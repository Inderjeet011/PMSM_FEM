#!/usr/bin/env python3
"""
Magnetostatic sanity check on the full 3D PMSM geometry.

Goal:
    Solve a pure magnetostatic problem with *only* permanent magnets:

        curl(nu curl A) = - curl M

    on the existing 3D mesh, using the same materials and magnetization
    as the time‑dependent A–V solver. This isolates whether the physics
    and AMS setup are capable of producing a realistic B field (~0.1–1 T)
    when the problem is simpler (no conductivity, no time stepping,
    no Schur complement).
"""

import time
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import ufl
from dolfinx import fem, io
from dolfinx.fem import petsc

from load_mesh import (
    SimulationConfig3D,
    load_mesh,
    setup_materials,
    setup_boundary_conditions,
    DomainTags3D,
)
from solve_equations import (
    setup_sources,
    initialise_magnetisation,
)
from compute_B import compute_B_field


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("\n=== MAGNETOSTATIC PM TEST (3D) ===")

    # ------------------------------------------------------------------
    # 1. Load mesh, materials, and spaces
    # ------------------------------------------------------------------
    config = SimulationConfig3D()

    mesh, ct, ft = load_mesh(config.mesh_path)
    if rank == 0:
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
        print(f"Mesh loaded: {num_cells} cells")

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    dx_magnets = None
    if ct is not None:
        # Measure over permanent magnet regions only
        def measure_over(dx_meas, markers):
            m = None
            for marker in markers:
                term = dx_meas(marker)
                m = term if m is None else m + term
            return m

        dx_magnets = measure_over(dx, DomainTags3D.MAGNETS)
    else:
        dx_magnets = ufl.dx(domain=mesh)

    # Materials (nu carries 1/(mu0*mu_r))
    sigma, nu, _density = setup_materials(mesh, ct, config)

    # Function spaces: same Nédélec space as main solver
    nedelec = basix.ufl.element("N1curl", mesh.basix_cell(), config.degree_A)
    A_space = fem.functionspace(mesh, nedelec)

    if rank == 0:
        ndofs_A = A_space.dofmap.index_map.size_global * A_space.dofmap.index_map_bs
        print(f"A_space DOFs: {ndofs_A}")

    # Boundary conditions: keep weak BC philosophy (no strong BC on A)
    # We only need the facet tags for post‑processing; no BCs applied here.
    _bc_A, _bc_V, _block_bcs = setup_boundary_conditions(
        mesh, ft, A_space, A_space  # V_space not used; dummy
    )

    # ------------------------------------------------------------------
    # 2. Sources: permanent magnets only (no currents, no conduction)
    # ------------------------------------------------------------------
    J_z, M_vec = setup_sources(mesh, ct)
    initialise_magnetisation(mesh, ct, M_vec, config)
    # Zero out currents: pure PM excitation
    J_z.x.array[:] = 0.0

    if rank == 0:
        M_array = M_vec.x.array.reshape((-1, 3))
        M_mag = np.linalg.norm(M_array, axis=1)
        print(
            f"Magnetization magnitude: max={np.max(M_mag):.3e} A/m, "
            f"mean={np.mean(M_mag):.3e} A/m"
        )

    # ------------------------------------------------------------------
    # 3. Build magnetostatic form: curl(nu curl A) with small mass shift
    # ------------------------------------------------------------------
    A = ufl.TrialFunction(A_space)
    v = ufl.TestFunction(A_space)

    curlA = ufl.curl(A)
    curlv = ufl.curl(v)

    # Operator: curl(nu curl A) + eps * A
    eps = fem.Constant(mesh, PETSc.ScalarType(1e-6))
    a_ms = ufl.inner(nu * curlA, curlv) * ufl.dx + eps * ufl.inner(A, v) * ufl.dx
    L_ms = -ufl.inner(M_vec, curlv) * dx_magnets

    a_form = fem.form(a_ms)
    L_form = fem.form(L_ms)

    if rank == 0:
        print("Assembling magnetostatic operator...")
    t0 = time.time()
    A_mat = petsc.assemble_matrix(a_form, bcs=None)
    A_mat.assemble()
    A_mat.setOption(PETSc.Mat.Option.SPD, True)
    t1 = time.time()
    if rank == 0:
        print(
            f"Magnetostatic matrix assembled: size={A_mat.getSize()}, "
            f"Frobenius norm={A_mat.norm(PETSc.NormType.NORM_FROBENIUS):.3e}, "
            f"time={t1-t0:.3f} s"
        )

    b_vec = petsc.create_vector(L_form)
    petsc.assemble_vector(b_vec, L_form)
    b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    if rank == 0:
        with b_vec.localForm() as b_local:
            b_arr = b_local.array_r
            print(
                f"RHS: ||b||={np.linalg.norm(b_arr):.3e}, "
                f"max|b|={np.max(np.abs(b_arr)):.3e}"
            )

    # ------------------------------------------------------------------
    # 4. Reference solve WITHOUT AMS (use AMG) to get expected |B|
    # ------------------------------------------------------------------
    if rank == 0:
        print("Configuring reference KSP (AMG, no AMS) for magnetostatic solve...")

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A_mat, A_mat)
    ksp.setType("cg")
    ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=500)

    pc = ksp.getPC()
    pc.setType("hypre")
    try:
        pc.setHYPREType("boomeramg")
    except Exception:
        # Fallback to simple Jacobi if AMG is not available
        pc.setType("jacobi")

    if rank == 0:
        print("Solving magnetostatic system with AMG (no AMS)...")

    A_sol = b_vec.duplicate()
    A_sol.set(0.0)

    t0 = time.time()
    ksp.solve(b_vec, A_sol)
    t1 = time.time()

    reason = ksp.getConvergedReason()
    its = ksp.getIterationNumber()
    res = ksp.getResidualNorm()
    if rank == 0:
        reason_map = {
            -3: "DIVERGED_ITS",
            -9: "DIVERGED_NANORINF",
            -11: "DIVERGED_BREAKDOWN",
            1: "CONVERGED_RTOL",
            2: "CONVERGED_ATOL",
            3: "CONVERGED_ITS",
        }
        print(
            f"Reference solve done in {t1-t0:.3f} s, "
            f"reason={reason_map.get(reason, reason)}, its={its}, residual={res:.3e}"
        )

    # ------------------------------------------------------------------
    # 5. Map solution back to Function and compute B field
    # ------------------------------------------------------------------
    A_fun = fem.Function(A_space, name="A")
    with A_sol.localForm() as src:
        A_fun.x.array[:] = src.array_r[: A_fun.x.array.size]
    A_fun.x.scatter_forward()

    if rank == 0:
        A_arr = A_fun.x.array.reshape((-1, 3))
        A_mag = np.linalg.norm(A_arr, axis=1)
        print(
            f"A field: ||A||={np.linalg.norm(A_arr):.3e} Wb/m, "
            f"max|A|={np.max(A_mag):.3e}, mean|A|={np.mean(A_mag):.3e}"
        )

    # B field via existing compute_B_field helper (full domain, debug)
    lagrange = basix.ufl.element("Lagrange", mesh.basix_cell(), config.degree_A)
    lagrange_vec = basix.ufl.element(
        "Lagrange", mesh.basix_cell(), config.degree_A, shape=(3,)
    )
    B_space = fem.functionspace(mesh, lagrange_vec)
    B_mag_space = fem.functionspace(mesh, lagrange)

    B_sol, B_mag_fun, max_B, min_B, norm_B, B_dg = compute_B_field(
        mesh,
        A_fun,
        B_space,
        B_mag_space,
        config,
        cell_tags=ct,
        debug=True,
        restrict_to_airgap=False,
    )

    if rank == 0:
        print(
            f"[RESULT] Magnetostatic B field: max|B|={max_B:.3e} T, "
            f"min|B|={min_B:.3e} T, ||B||={norm_B:.3e}"
        )

        # Optional: write to a separate XDMF for inspection
        results_path = (
            config.results_path.parent / "magnetostatic_pm_test.xdmf"
        )
        print(f"Writing magnetostatic fields to: {results_path}")
        with io.XDMFFile(comm, str(results_path), "w") as xdmf:
            xdmf.write_mesh(mesh)
            if ct is not None:
                xdmf.write_meshtags(ct, mesh.geometry)
            if ft is not None:
                xdmf.write_meshtags(ft, mesh.geometry)
            B_dg.name = "B_dg"
            B_sol.name = "B"
            B_mag_fun.name = "B_Magnitude"
            xdmf.write_function(B_dg)
            xdmf.write_function(B_sol)
            xdmf.write_function(B_mag_fun)

        print("=== MAGNETOSTATIC PM TEST DONE ===\n")


if __name__ == "__main__":
    main()


