import argparse
from pathlib import Path
from datetime import datetime
from typing import Union, Dict
import time
import dolfinx.fem.petsc as _petsc
import dolfinx.mesh
import numpy as np
try:
    import tqdm
except ImportError:
    tqdm = None
import basix.ufl
import ufl
from dolfinx import cpp, fem, io, default_scalar_type
from mpi4py import MPI
from petsc4py import PETSc

class MagneticField2D():
    def __init__(self, AzV: fem.Function, form_compiler_options: dict = {}, jit_parameters: dict = {}):
        degree = AzV.function_space.ufl_element().degree
        mesh = AzV.function_space.mesh
        cell = mesh.ufl_cell()
        el_B = basix.ufl.element("DG", cell.cellname(), max(degree - 1, 1), shape=(mesh.geometry.dim,))
        VB = fem.functionspace(mesh, el_B)
        self.B = fem.Function(VB)
        B_2D = ufl.as_vector((AzV[0].dx(1), -AzV[0].dx(0)))
        self.Bexpr = fem.Expression(B_2D, VB.element.interpolation_points,
                                    form_compiler_options=form_compiler_options,
                                    jit_options=jit_parameters)
    def interpolate(self):
        self.B.interpolate(self.Bexpr)

def update_current_density(J_0: fem.Function, omega: float, t: float, ct: cpp.mesh.MeshTags_int32,
                           currents: Dict[np.int32, Dict[str, float]]):
    J = 1413810.0970277672
    J_0.x.array[:] = 0
    for domain, values in currents.items():
        _cells = ct.find(domain)
        J_0.x.array[_cells] = np.full(len(_cells), J * values["alpha"] * np.cos(omega * t + values["beta"]))

def update_magnetization(Mvec, coercivity, omega_u, t, ct, domains, pm_orientation):
    block_size = 2
    coercivity = 8.38e5
    sign = 1
    for (material, domain) in domains.items():
        if material == 'PM':
            for marker in domain:
                inout = 1 if marker in [13, 15, 17, 19, 21] else -1
                angle = pm_orientation[marker] + omega_u * t
                Mx = coercivity * np.cos(angle) * sign * inout
                My = coercivity * np.sin(angle) * sign * inout
                cells = ct.find(marker)
                for cell in cells:
                    idx = block_size * cell
                    Mvec.x.array[idx + 0] = Mx
                    Mvec.x.array[idx + 1] = My
    Mvec.x.scatter_forward()

def solve_pmsm(outdir: Path = Path("results"), progress: bool = False, save_output: bool = False):
    fname = Path("meshes") / "three_phase"
    omega_u: np.float64 = 62.83
    degree: np.int32 = 1
    apply_torque: bool = False
    form_compiler_options: dict = {}
    jit_parameters: dict = {}
    mu_0 = 1.25663753e-6
    model_parameters = {
        "mu_0": 1.25663753e-6,
        "freq": 50,
        "J": 1413810.0970277672,
        "mu": {"Cu": 0.999991*mu_0, "Stator": 100*mu_0, "Rotor": 100*mu_0, "Al": 100*mu_0, "Air": mu_0, "AirGap": mu_0, "PM": 1.04457*mu_0},
        "sigma": {"Rotor": 2e6, "Al": 2e6, "Stator": 0, "Cu": 0, "Air": 0, "AirGap": 0, "PM": 6.25e5},
        "densities": {"Rotor": 7850, "Al": 7850, "Stator": 0, "Air": 0, "Cu": 0, "AirGap": 0, "PM": 7500}
    }
    
    freq = model_parameters["freq"]
    T = 0.04  # Shorter simulation: 2 periods at 50Hz
    dt_ = 0.002  # Slightly larger timestep for faster run
    omega_J = 2 * np.pi * freq

    domains = {"Air": (1,), "AirGap": (2, 3), "Al": (4,), "Rotor": (5, ), 
               "Stator": (6, ), "Cu": (7, 8, 9, 10, 11, 12),
               "PM": (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)}
    
    currents = {7: {"alpha": 1, "beta": 0}, 8: {"alpha": -1, "beta": 2 * np.pi / 3},
                9: {"alpha": 1, "beta": 4 * np.pi / 3}, 10: {"alpha": -1, "beta": 0},
                11: {"alpha": 1, "beta": 2 * np.pi / 3},
                12: {"alpha": -1, "beta": 4 * np.pi / 3}}

    surface_map: Dict[str, Union[int, str]] = {"Exterior": 1, "MidAir": 2, "restriction": "+"}

    with io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        ct = xdmf.read_meshtags(mesh, name="Cell_markers")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        ft = xdmf.read_meshtags(mesh, name="Facet_markers")

    DG0 = fem.functionspace(mesh, ("DG", 0))
    mu = fem.Function(DG0)
    sigma = fem.Function(DG0)
    density = fem.Function(DG0)
    for (material, domain) in domains.items():
        for marker in domain:
            cells = ct.find(marker)
            mu.x.array[cells] = model_parameters["mu"][material]
            sigma.x.array[cells] = model_parameters["sigma"][material]
            density.x.array[cells] = model_parameters["densities"][material]

    cell = mesh.ufl_cell()
    FE = basix.ufl.element("Lagrange", cell.cellname(), degree)
    ME = basix.ufl.mixed_element([FE, FE])
    VQ = fem.functionspace(mesh, ME)

    Az, V = ufl.TrialFunctions(VQ)
    vz, q = ufl.TestFunctions(VQ)
    AnVn = fem.Function(VQ)
    An, _ = ufl.split(AnVn)
    J0z = fem.Function(DG0)

    Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
    Omega_c = domains["Rotor"] + domains["Al"] + domains["PM"]
    Omega_pm = domains["PM"]

    coercivity = 8.38e5
    DG0v = fem.functionspace(mesh, ("DG", 0, (2,)))
    Mvec = fem.Function(DG0v)

    pm_spacing = (np.pi / 6) + (np.pi / 30)
    pm_angles = np.asarray([i * pm_spacing for i in range(10)], dtype=np.float64)

    pm_orientation = {}
    for i, pm_marker in enumerate(Omega_pm):
        pm_orientation[pm_marker] = pm_angles[i]

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    dt = fem.Constant(mesh, dt_)
    x = ufl.SpatialCoordinate(mesh)
    omega = fem.Constant(mesh, default_scalar_type(omega_u))

    u = omega * ufl.as_vector((-x[1], x[0]))
    curl_vz = ufl.as_vector((vz.dx(1), -vz.dx(0)))
    mag_term = (mu_0/mu) * ufl.inner(Mvec, curl_vz) * dx(Omega_pm)

    f_a = dt / mu * ufl.inner(ufl.grad(Az), ufl.grad(vz)) * dx(Omega_n + Omega_c) \
          + sigma * (Az - An) * vz * dx(Omega_c) \
          + dt * sigma * ufl.dot(u, ufl.grad(Az)) * vz * dx(Omega_c) \
          - dt * J0z * vz * dx(Omega_n) \
          - dt * mag_term
    f_v = dt * sigma * (V.dx(0) * q.dx(0) + V.dx(1) * q.dx(1)) * dx(Omega_c)
    form_av = f_a + f_v
    a, L = ufl.system(form_av)

    cells_n = np.hstack([ct.find(domain) for domain in Omega_n])
    Q, _ = VQ.sub(1).collapse()
    mesh.topology.create_connectivity(tdim, tdim)
    deac_dofs = fem.locate_dofs_topological((VQ.sub(1), Q), tdim, cells_n)

    zeroQ = fem.Function(Q)
    zeroQ.x.array[:] = 0
    bc_Q = fem.dirichletbc(zeroQ, deac_dofs, VQ.sub(1))

    V_, _ = VQ.sub(0).collapse()
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    bndry_dofs = fem.locate_dofs_topological((VQ.sub(0), V_), tdim - 1, boundary_facets)
    zeroV = fem.Function(V_)
    zeroV.x.array[:] = 0
    bc_V = fem.dirichletbc(zeroV, bndry_dofs, VQ.sub(0))
    bcs = [bc_V, bc_Q]

    cpp_a = fem.form(a, form_compiler_options=form_compiler_options, jit_options=jit_parameters)
    pattern = fem.create_sparsity_pattern(cpp_a)
    block_size = VQ.dofmap.index_map_bs
    deac_blocks = deac_dofs[0] // block_size
    pattern.insert_diagonal(deac_blocks)
    pattern.finalize()

    A = cpp.la.petsc.create_matrix(mesh.comm, pattern)
    A.zeroEntries()
    if not apply_torque:
        A.zeroEntries()
        _petsc.assemble_matrix(A, cpp_a, bcs=bcs)
        A.assemble()

    cpp_L = fem.form(L, form_compiler_options=form_compiler_options, jit_options=jit_parameters)
    b = _petsc.create_vector(cpp_L)

    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver_prefix = f"PMSM_solve_{id(solver)}"
    solver.setOptionsPrefix(solver_prefix)

    opts = PETSc.Options()
    opts.prefixPush(solver_prefix)
    petsc_options: dict = {"ksp_type": "preonly", "pc_type": "lu"}
    for k, v in petsc_options.items():
        opts[k] = v
    opts.prefixPop()
    solver.setFromOptions()

    AzV = fem.Function(VQ)
    Az_out = AzV.sub(0).collapse()
    V_out = AzV.sub(1).collapse()
    post_B = MagneticField2D(AzV)

    V_B_CG = fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    B_CG = fem.Function(V_B_CG, name="B")
    Az_out.name = "Az"
    V_out.name = "V"

    if save_output:
        xdmf_Az = io.XDMFFile(mesh.comm, str(outdir / "Az.xdmf"), "w")
        xdmf_B = io.XDMFFile(mesh.comm, str(outdir / "B.xdmf"), "w")
        xdmf_V = io.XDMFFile(mesh.comm, str(outdir / "V.xdmf"), "w")

    x = ufl.SpatialCoordinate(mesh)
    r = ufl.sqrt(x[0]**2 + x[1]**2)
    L = 1
    I_rotor = mesh.comm.allreduce(fem.assemble_scalar(fem.form(L * r**2 * density * dx(Omega_c))))

    num_steps = int(T / float(dt.value))
    t = 0.
    update_current_density(J0z, omega_J, t, ct, currents)
    update_magnetization(Mvec, coercivity, omega_u, t, ct, domains, pm_orientation)
    
    if MPI.COMM_WORLD.rank == 0 and progress and tqdm is not None:
        progressbar = tqdm.tqdm(desc="Solving PMSM", total=num_steps)

    for i in range(num_steps):
        if MPI.COMM_WORLD.rank == 0 and progress and tqdm is not None:
            progressbar.update(1)
        t += float(dt.value)
        update_current_density(J0z, omega_J, t, ct, currents)
        update_magnetization(Mvec, coercivity, omega_u, t, ct, domains, pm_orientation)

        with b.localForm() as loc_b:
            loc_b.set(0)
        _petsc.assemble_vector(b, cpp_L)
        _petsc.apply_lifting(b, [cpp_a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b, bcs)

        solver.solve(b, AzV.x.petsc_vec)
        AzV.x.scatter_forward()
    
        AnVn.x.array[:] = AzV.x.array
        AnVn.x.scatter_forward()
        
        if MPI.COMM_WORLD.rank == 0:
            Az_min, Az_max = min(AzV.sub(0).collapse().x.array[:]), max(AzV.sub(0).collapse().x.array[:])
            B_min, B_max = min(post_B.B.x.array[:]), max(post_B.B.x.array[:])
            print(f"t={t:.4f}s: Az=[{Az_min:.6e}, {Az_max:.6e}], B=[{B_min:.6e}, {B_max:.6e}]")
        
        if save_output:
            post_B.interpolate()
            Az_out.x.array[:] = AzV.sub(0).collapse().x.array[:]
            V_out.x.array[:] = AzV.sub(1).collapse().x.array[:]
            
            # Project DG B to CG B for XDMF compatibility
            B_CG.interpolate(post_B.B)
            
            # Write mesh on first step only
            if i == 0:
                xdmf_Az.write_mesh(mesh)
                xdmf_V.write_mesh(mesh)
                xdmf_B.write_mesh(mesh)
            
            xdmf_Az.write_function(Az_out, t)
            xdmf_V.write_function(V_out, t)
            xdmf_B.write_function(B_CG, t)
    
    b.destroy()
    if save_output:
        xdmf_Az.close()
        xdmf_B.close()
        xdmf_V.close()

    elements = mesh.topology.index_map(mesh.topology.dim).size_global
    num_dofs = VQ.dofmap.index_map.size_global * VQ.dofmap.index_map_bs
    if mesh.comm.rank == 0:
        print(f"\n✅ Simulation complete: {elements} elements, {num_dofs} DOFs")


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="PMSM solver with XDMF output")
    parser.add_argument('--progress', action='store_true', help="Show progress bar", default=False)
    parser.add_argument('--output', action='store_true', help="Save XDMF output", default=False)

    args = parser.parse_args()
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%b_%d_%H_%M_%S")
    outdir = Path(f"XDMF_{formatted_datetime}")
    outdir.mkdir(exist_ok=True)
    print(f"="*70)
    print(f" PMSM SOLVER")
    print(f" Saving to {outdir}")
    print(f"="*70)
    solve_pmsm(outdir=outdir, progress=args.progress, save_output=args.output)
    
    end_time = time.time()
    print(f"\n⏱️  Elapsed time: {end_time - start_time:.2f}s")
    print(f"📁 Results in: {outdir}/")
    print(f"="*70)

