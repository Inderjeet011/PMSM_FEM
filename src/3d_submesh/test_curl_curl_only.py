"""
Test pure curl-curl A equation: nu*curl(A)·curl(v) = M·curl(v) in PM.
No sigma, no motion, no regularization, no A-V coupling.
Checks if the simple formulation converges and gives reasonable B.

Run: cd src/3d_submesh && python test_curl_curl_only.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "3d"))
from load_mesh import omega_rs, omega_rpm, omega_c, omega_pm, MAGNETS

from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl
import basix.ufl

from load_mesh_submesh import load_mesh_and_extract_submesh, setup_materials, conducting
from solver_utils_submesh import make_config, measure_over, setup_sources, initialise_magnetisation, rotate_magnetization
from load_mesh_submesh import setup_boundary_conditions_parent


def main():
    config = make_config()
    mesh_parent, mesh_conductor, cell_tags_parent, cell_tags_conductor, facet_tags_parent, conductor_cells, entity_map = load_mesh_and_extract_submesh(config.mesh_path)

    dx_parent = ufl.Measure("dx", domain=mesh_parent, subdomain_data=cell_tags_parent)
    dx_pm = measure_over(dx_parent, omega_pm())

    sigma, nu, density = setup_materials(mesh_parent, cell_tags_parent, config)
    J_z, M_vec = setup_sources(mesh_parent)
    initialise_magnetisation(mesh_parent, cell_tags_parent, M_vec, config)
    t = config.dt
    rotate_magnetization(cell_tags_parent, M_vec, config, t)

    A_space = fem.functionspace(mesh_parent, basix.ufl.element("N1curl", mesh_parent.basix_cell(), config.degree_A))
    bc_A = setup_boundary_conditions_parent(mesh_parent, facet_tags_parent, A_space)

    # Pure curl-curl: a00 = nu*curl(A)·curl(v) [+ tiny eps for PC], L0 = nu*mu0*M·curl(v)
    A = ufl.TrialFunction(A_space)
    v = ufl.TestFunction(A_space)
    curlA = ufl.curl(A)
    curlv = ufl.curl(v)
    mu0 = config.mu0
    eps = fem.Constant(mesh_parent, PETSc.ScalarType(1e-6))  # needed for AMS; LU works with any
    use_direct = True   # LU works; AMS fails (reason=-9) or returns wrong solution

    a00 = nu * ufl.inner(curlA, curlv) * dx_parent + eps * ufl.inner(A, v) * dx_parent
    L0 = ufl.inner(nu * mu0 * M_vec, curlv) * dx_pm

    a_form = fem.form(a00)
    L_form = fem.form(L0)

    # Assemble
    A_mat = petsc.assemble_matrix(a_form, bcs=[bc_A])
    A_mat.assemble()
    b_vec = petsc.create_vector(A_space)
    petsc.assemble_vector(b_vec, L_form)
    petsc.set_bc(b_vec, [bc_A])
    b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Solver: CG or GMRES + AMS
    from dolfinx.cpp.fem.petsc import discrete_gradient
    V_ams = fem.functionspace(mesh_parent, ("Lagrange", 1))
    G = discrete_gradient(V_ams._cpp_object, A_space._cpp_object)
    G.assemble()
    xcoord = ufl.SpatialCoordinate(mesh_parent)
    verts = []
    for dim in range(3):
        f = fem.Function(V_ams)
        f.interpolate(fem.Expression(xcoord[dim], V_ams.element.interpolation_points))
        f.x.scatter_forward()
        verts.append(f.x.petsc_vec)
    e0, e1, e2 = G.createVecLeft(), G.createVecLeft(), G.createVecLeft()
    G.mult(verts[0], e0)
    G.mult(verts[1], e1)
    G.mult(verts[2], e2)

    # Null space for curl-curl (gradient): needed for AMS
    nullsp = PETSc.NullSpace().create(vectors=[e0, e1, e2], comm=mesh_parent.comm)
    A_mat.setNullSpace(nullsp)
    A_mat.setTransposeNullSpace(nullsp)

    ksp = PETSc.KSP().create(mesh_parent.comm)
    ksp.setOperators(A_mat, A_mat)
    if use_direct:
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
    else:
        ksp.setType("gmres")
        ksp.setGMRESRestart(200)
        ksp.setTolerances(rtol=1e-6, atol=0.0, max_it=500)
        pc = ksp.getPC()
        pc.setType("hypre")
        pc.setHYPREType("ams")
        pc.setHYPREDiscreteGradient(G)
        pc.setHYPRESetEdgeConstantVectors(e0, e1, e2)

    x = b_vec.duplicate()
    x.set(0.0)
    ksp.solve(b_vec, x)

    its = ksp.getIterationNumber()
    reason = ksp.getConvergedReason()
    converged = reason > 0

    # Residual
    r = b_vec.duplicate()
    A_mat.mult(x, r)
    r.axpy(-1.0, b_vec)
    rnorm = r.norm(PETSc.NormType.NORM_2)
    bnorm = b_vec.norm(PETSc.NormType.NORM_2)
    rel_res = rnorm / bnorm if bnorm > 1e-30 else float("inf")

    # B = curl(A)
    A_sol = fem.Function(A_space)
    A_sol.x.array[:] = x.getArray(readonly=True)[:A_sol.x.array.size]
    A_sol.x.scatter_forward()

    curlA_sol = ufl.curl(A_sol)
    DG_vec = fem.functionspace(mesh_parent, ("DG", 0, (3,)))
    B_dg = fem.Function(DG_vec)
    B_dg.interpolate(fem.Expression(curlA_sol, DG_vec.element.interpolation_points))
    B_dg.x.scatter_forward()
    B_mag = np.linalg.norm(B_dg.x.array.reshape(-1, 3), axis=1)
    pm_cells = np.concatenate([cell_tags_parent.find(m) for m in MAGNETS])
    pm_cells = np.unique(pm_cells)

    if mesh_parent.comm.rank == 0:
        print("\n" + "=" * 60)
        print("PURE CURL-CURL TEST (nu*curl*curl = M·curl, no sigma/motion/reg)")
        print("=" * 60)
        print(f"  Solver: its={its}, reason={reason}  Converged: {converged}")
        print(f"  ||b-Ax||: {rnorm:.6e}  ||b||: {bnorm:.6e}  rel_res: {rel_res:.4e}")
        print(f"  Max |B| global: {np.max(B_mag):.4e} T")
        print(f"  Max |B| in PM:  {np.max(B_mag[pm_cells]) if pm_cells.size > 0 else 0:.4e} T")
        print(f"  Expected B in PM ~ {config.magnet_remanence} T")
        print("=" * 60)


if __name__ == "__main__":
    main()
