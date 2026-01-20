"""Small helper functions for the 3D solver."""

import numpy as np
from pathlib import Path
from types import SimpleNamespace
from mpi4py import MPI

from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl

from load_mesh import CURRENT_MAP, MAGNETS, AIR_GAP


def make_config():
    from mesh_3D import model_parameters

    freq = float(model_parameters["freq"])
    pole_pairs = 5  # number of pole pairs (10 poles => 5 pole pairs)
    # Time resolution: number of timesteps per *electrical* period (1/freq).
    # 3 is very coarse; use a higher default for better transient accuracy.
    steps_per_period = 40
    dt = (1.0 / freq) / steps_per_period

    omega_e = 2.0 * np.pi * freq
    # Rotation direction convention:
    # +omega_m => CCW in the x–y plane when viewed from +z (right-hand rule).
    # Set to -1.0 to reverse rotation (CW when viewed from +z).
    rotation_direction = -1.0
    omega_m = rotation_direction * (omega_e / max(pole_pairs, 1))

    root = Path(__file__).parents[2]
    return SimpleNamespace(
        dt=dt,
        # Run a few electrical periods by default (helps reach periodic behavior).
        num_steps=1,  # Run 10 time steps to see torque evolution
        degree_A=1,
        degree_V=1,
        mu0=float(model_parameters["mu_0"]),
        coil_current_peak=float(model_parameters["J"]),
        magnet_remanence=1.2,
        omega_e=omega_e,
        omega_m=omega_m,
        mesh_path=root / "meshes" / "3d" / "pmesh3D_ipm.xdmf",
        results_path=root / "results" / "3d" / "av_solver.xdmf",
        write_results=True,
        # PETSc iteration limits: bump these if you see non-convergence
        outer_max_it=100,
        ksp_A_max_it=5,
        # Regularization on the actual A00 block (adds epsilon * (A,v) over all domains)
        # Start small; increase if you see stagnation.
        epsilon_A=1e-8,
        # Keep existing preconditioner regularization
        epsilon_A_spd=1e-6,
    )


def measure_over(dx, markers):
    measure = None
    for marker in markers:
        term = dx(marker)
        measure = term if measure is None else measure + term
    return measure


def make_ground_bc_V(V_space, cell_tags, conductor_markers):
    for m in conductor_markers:
        cells = cell_tags.find(m)
        if cells.size == 0:
            continue
        dofs = V_space.dofmap.cell_dofs(int(cells[0]))
        if len(dofs) == 0:
            continue
        u0 = fem.Function(V_space)
        u0.x.array[:] = 0.0
        return fem.dirichletbc(u0, np.array([int(dofs[0])], dtype=np.int32))
    u0 = fem.Function(V_space)
    u0.x.array[:] = 0.0
    return fem.dirichletbc(u0, np.array([], dtype=np.int32))


def setup_sources(mesh):
    DG0 = fem.functionspace(mesh, ("DG", 0))
    DG0_vec = fem.functionspace(mesh, ("DG", 0, (3,)))
    return fem.Function(DG0, name="Jz"), fem.Function(DG0_vec, name="M")


def initialise_magnetisation(mesh, cell_tags, M_vec, config):
    dofmap = mesh.geometry.dofmap
    coords = mesh.geometry.x
    vec_view = M_vec.x.array.reshape((-1, 3))
    magnitude = config.magnet_remanence / max(config.mu0, 1e-12)
    for marker in MAGNETS:
        cells = cell_tags.find(marker)
        for c in cells:
            geom_dofs = dofmap[c]
            cell_coords = coords[geom_dofs]
            cx = float(np.mean(cell_coords[:, 0]))
            cy = float(np.mean(cell_coords[:, 1]))
            norm = np.hypot(cx, cy)
            direction = np.array([1.0, 0.0, 0.0]) if norm < 1e-12 else np.array([cx / norm, cy / norm, 0.0])
            vec_view[c, :] = magnitude * direction
    M_vec.x.scatter_forward()


def rotate_magnetization(cell_tags, M_vec, config, t):
    theta_rot = config.omega_m * t
    vec_view = M_vec.x.array.reshape((-1, 3))
    magnitude = config.magnet_remanence / max(config.mu0, 1e-12)
    pm_spacing = (np.pi / 6) + (np.pi / 30)
    pm_angles = np.asarray([i * pm_spacing for i in range(10)])
    for marker in MAGNETS:
        cells = cell_tags.find(marker)
        sign = 1 if ((marker - 13) % 2 == 0) else -1
        theta_now = pm_angles[marker - 13] + theta_rot
        for c in cells:
            vec_view[c, 0] = sign * magnitude * np.cos(theta_now)
            vec_view[c, 1] = sign * magnitude * np.sin(theta_now)
            vec_view[c, 2] = 0.0
    M_vec.x.scatter_forward()


def update_currents(cell_tags, J_z, config, t):
    omega = config.omega_e
    J_peak = config.coil_current_peak
    J_z.x.array[:] = 0.0
    for marker, meta in CURRENT_MAP.items():
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        J_z.x.array[cells] = J_peak * meta["alpha"] * np.sin(omega * t + meta["beta"])


def assemble_rhs(a_blocks, L_blocks, block_bcs, A_space, V_space):
    bA = petsc.create_vector(A_space)
    petsc.assemble_vector(bA, L_blocks[0])
    petsc.apply_lifting(bA, [a_blocks[0][1]], bcs=[block_bcs[1]])
    petsc.set_bc(bA, block_bcs[0])
    bA.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bV = petsc.create_vector(V_space)
    petsc.assemble_vector(bV, L_blocks[1])
    petsc.apply_lifting(bV, [a_blocks[1][0]], bcs=[block_bcs[0]])
    petsc.set_bc(bV, block_bcs[1])
    bV.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    return PETSc.Vec().createNest([bA, bV])


def solve_one_step(mesh, A_space, V_space, cell_tags, config, ksp, mat_nest, a_blocks,
                   L_blocks, block_bcs, J_z, M_vec, A_prev, t):
    update_currents(cell_tags, J_z, config, t)
    rotate_magnetization(cell_tags, M_vec, config, t)

    rhs = assemble_rhs(a_blocks, L_blocks, block_bcs, A_space, V_space)
    sol = PETSc.Vec().createNest([petsc.create_vector(A_space), petsc.create_vector(V_space)], comm=mesh.comm)
    sol.set(0.0)
    ksp.solve(rhs, sol)

    A_sol = fem.Function(A_space, name="A")
    V_sol = fem.Function(V_space, name="V")
    with sol.getNestSubVecs()[0].localForm() as src:
        A_sol.x.array[:] = src.array_r[:A_sol.x.array.size]
    A_sol.x.scatter_forward()
    with sol.getNestSubVecs()[1].localForm() as src:
        V_sol.x.array[:] = src.array_r[:V_sol.x.array.size]
    V_sol.x.scatter_forward()

    A_prev.x.array[:] = A_sol.x.array[:]
    return A_sol, V_sol


def compute_B_field(mesh, A_sol, B_space, B_magnitude_space):
    curlA = ufl.curl(A_sol)

    DG_vec = fem.functionspace(mesh, ("DG", 0, (3,)))
    B_dg = fem.Function(DG_vec, name="B_dg")
    B_dg.interpolate(fem.Expression(curlA, DG_vec.element.interpolation_points))
    B_dg.x.scatter_forward()

    B_sol = fem.Function(B_space, name="B")
    B_sol.interpolate(fem.Expression(curlA, B_space.element.interpolation_points))
    B_sol.x.scatter_forward()

    B_mag = fem.Function(B_magnitude_space, name="B_Magnitude")
    B_mag.interpolate(
        fem.Expression(
            ufl.sqrt(ufl.inner(curlA, curlA)),
            B_magnitude_space.element.interpolation_points,
        )
    )
    B_mag.x.scatter_forward()

    vals = B_mag.x.array
    return B_sol, B_mag, float(vals.max()), float(vals.min()), float(np.linalg.norm(vals)), B_dg


def compute_B_in_airgap(mesh, B_dg, cell_tags, comm=None, B_sol=None):
    """
    Compute B-field statistics in the airgap region.
    
    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        The mesh
    B_dg : dolfinx.fem.Function
        DG0 vector function containing B field (B = curl(A))
    cell_tags : dolfinx.mesh.MeshTags
        Cell tags identifying different regions
    comm : mpi4py.MPI.Comm, optional
        MPI communicator (defaults to mesh.comm)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'B_avg': Average B magnitude in airgap (T)
        - 'B_max': Maximum B magnitude in airgap (T)
        - 'B_min': Minimum B magnitude in airgap (T)
        - 'B_rms': RMS B magnitude in airgap (T)
        - 'B_avg_vector': Average B vector [Bx, By, Bz] in airgap (T)
        - 'n_cells': Number of airgap cells
        - 'B_radial_avg': Average radial component of B (T)
        - 'B_tangential_avg': Average tangential component of B (T)
    """
    if comm is None:
        comm = mesh.comm
    
    # Find all airgap cells (tags 2 and 3)
    airgap_cells = np.array([], dtype=np.int32)
    for tag in AIR_GAP:
        cells = cell_tags.find(tag)
        if cells.size > 0:
            airgap_cells = np.concatenate([airgap_cells, cells])
    
    if airgap_cells.size == 0:
        if comm.rank == 0:
            print("Warning: No airgap cells found")
        return {
            'B_avg': 0.0, 'B_max': 0.0, 'B_min': 0.0, 'B_rms': 0.0,
            'B_avg_vector': np.array([0.0, 0.0, 0.0]),
            'n_cells': 0, 'B_radial_avg': 0.0, 'B_tangential_avg': 0.0
        }
    
    # Get B values in airgap cells
    # Try using B_sol (interpolated Lagrange) if available - it's smoother and may be more accurate
    # Otherwise use B_dg (DG0 cell-wise values)
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    coords = mesh.geometry.x
    dofmap = mesh.geometry.dofmap
    centers = coords[dofmap].mean(axis=1)
    airgap_centers = centers[airgap_cells]
    
    n_cells_total = mesh.topology.index_map(3).size_local
    
    if B_sol is not None:
        # Method 1: Use B_sol (interpolated) - evaluate at cell centers
        # This gives smoother, more accurate values for thin regions
        try:
            from dolfinx import geometry
            bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
            cell_candidates = geometry.compute_collisions_points(bb_tree, airgap_centers)
            colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, airgap_centers)
            
            B_airgap = np.zeros((len(airgap_cells), 3))
            for idx in range(len(airgap_cells)):
                center = airgap_centers[idx].reshape(1, -1)
                # Get the cell that contains this point
                cells_for_point = colliding_cells.links(idx)
                if len(cells_for_point) > 0:
                    cell = cells_for_point[0]
                    B_airgap[idx] = B_sol.eval(center, [cell])[0]
                else:
                    # Fallback to B_dg if point not found
                    cell = airgap_cells[idx]
                    if 3*cell+2 < B_dg.x.array.size:
                        B_airgap[idx] = B_dg.x.array[3*cell:3*cell+3]
        except Exception:
            # If B_sol evaluation fails, fall back to B_dg
            if len(airgap_cells) > 0 and 3 * max(airgap_cells) + 2 < B_dg.x.array.size:
                B_all = B_dg.x.array[:n_cells_total*3].reshape(n_cells_total, 3)
                B_airgap = B_all[airgap_cells]
            else:
                B_airgap = np.zeros((len(airgap_cells), 3))
                for idx, cell in enumerate(airgap_cells):
                    if 3*cell+2 < B_dg.x.array.size:
                        B_airgap[idx] = B_dg.x.array[3*cell:3*cell+3]
    else:
        # Method 2: Use B_dg (DG0 cell-wise values) - direct cell data
        if len(airgap_cells) > 0 and 3 * max(airgap_cells) + 2 < B_dg.x.array.size:
            B_all = B_dg.x.array[:n_cells_total*3].reshape(n_cells_total, 3)
            B_airgap = B_all[airgap_cells]
        else:
            # Fallback: direct indexing
            B_airgap = np.zeros((len(airgap_cells), 3))
            for idx, cell in enumerate(airgap_cells):
                if 3*cell+2 < B_dg.x.array.size:
                    B_airgap[idx] = B_dg.x.array[3*cell:3*cell+3]
    
    # Compute B magnitude for each cell
    B_mag_airgap = np.linalg.norm(B_airgap, axis=1)
    
    # Statistics
    B_avg = float(np.mean(B_mag_airgap))
    B_max = float(np.max(B_mag_airgap))
    B_min = float(np.min(B_mag_airgap))
    B_rms = float(np.sqrt(np.mean(B_mag_airgap**2)))
    
    # Average B vector
    B_avg_vector = np.mean(B_airgap, axis=0)
    
    # Compute radial and tangential components
    # Get cell centers for airgap cells
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    coords = mesh.geometry.x
    dofmap = mesh.geometry.dofmap
    centers = coords[dofmap].mean(axis=1)
    airgap_centers = centers[airgap_cells]
    
    # Convert to cylindrical coordinates
    r = np.linalg.norm(airgap_centers[:, :2], axis=1)
    theta = np.arctan2(airgap_centers[:, 1], airgap_centers[:, 0])
    
    # Unit vectors in radial and tangential directions
    e_r = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    e_theta = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
    
    # Project B onto radial and tangential directions
    B_radial = np.sum(B_airgap * e_r, axis=1)
    B_tangential = np.sum(B_airgap * e_theta, axis=1)
    
    B_radial_avg = float(np.mean(B_radial))
    B_tangential_avg = float(np.mean(B_tangential))
    
    # Aggregate across MPI ranks
    n_cells_local = len(airgap_cells)
    n_cells_global = comm.allreduce(n_cells_local, op=MPI.SUM)
    
    # For statistics, we need to gather values from all ranks
    # Simple approach: compute local stats, then reduce
    stats_local = np.array([B_avg * n_cells_local, B_max, B_min, 
                           B_rms**2 * n_cells_local, n_cells_local])
    stats_global = np.zeros_like(stats_local)
    comm.Allreduce(stats_local, stats_global, op=MPI.SUM)
    
    if n_cells_global > 0:
        B_avg_global = stats_global[0] / stats_global[4]
        B_max_global = comm.allreduce(B_max, op=MPI.MAX)
        B_min_global = comm.allreduce(B_min, op=MPI.MIN)
        B_rms_global = np.sqrt(stats_global[3] / stats_global[4])
    else:
        B_avg_global = 0.0
        B_max_global = 0.0
        B_min_global = 0.0
        B_rms_global = 0.0
    
    # For vector average, we need weighted sum
    B_vec_sum_local = B_avg_vector * n_cells_local
    B_vec_sum_global = np.zeros(3)
    comm.Allreduce(B_vec_sum_local, B_vec_sum_global, op=MPI.SUM)
    B_avg_vector_global = B_vec_sum_global / n_cells_global if n_cells_global > 0 else np.zeros(3)
    
    # Radial and tangential averages
    B_radial_sum_local = B_radial_avg * n_cells_local
    B_tangential_sum_local = B_tangential_avg * n_cells_local
    B_radial_sum_global = comm.allreduce(B_radial_sum_local, op=MPI.SUM)
    B_tangential_sum_global = comm.allreduce(B_tangential_sum_local, op=MPI.SUM)
    B_radial_avg_global = B_radial_sum_global / n_cells_global if n_cells_global > 0 else 0.0
    B_tangential_avg_global = B_tangential_sum_global / n_cells_global if n_cells_global > 0 else 0.0
    
    # Debug: Also compute B in magnets and rotor for comparison
    from load_mesh import MAGNETS, ROTOR
    B_magnet_avg = 0.0
    B_rotor_avg = 0.0
    airgap_radii_info = None
    
    # Get B in magnet and rotor cells for comparison (on all ranks)
    magnet_cells = np.array([], dtype=np.int32)
    for tag in MAGNETS:
        cells = cell_tags.find(tag)
        if cells.size > 0:
            magnet_cells = np.concatenate([magnet_cells, cells])
    
    rotor_cells = cell_tags.find(ROTOR[0])
    
    if len(magnet_cells) > 0 and len(rotor_cells) > 0 and len(airgap_cells) > 0:
        B_all = B_dg.x.array[:n_cells_total*3].reshape(n_cells_total, 3)
        B_magnet = B_all[magnet_cells]
        B_rotor = B_all[rotor_cells]
        
        # Get radii for airgap cells to verify they're in the right region
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        coords = mesh.geometry.x
        dofmap = mesh.geometry.dofmap
        centers = coords[dofmap].mean(axis=1)
        airgap_centers_local = centers[airgap_cells]
        airgap_radii_local = np.linalg.norm(airgap_centers_local[:, :2], axis=1)
        
        # Aggregate statistics
        n_magnet_local = len(magnet_cells)
        n_rotor_local = len(rotor_cells)
        B_magnet_sum_local = np.sum(np.linalg.norm(B_magnet, axis=1))
        B_rotor_sum_local = np.sum(np.linalg.norm(B_rotor, axis=1))
        
        n_magnet_global = comm.allreduce(n_magnet_local, op=MPI.SUM)
        n_rotor_global = comm.allreduce(n_rotor_local, op=MPI.SUM)
        B_magnet_sum_global = comm.allreduce(B_magnet_sum_local, op=MPI.SUM)
        B_rotor_sum_global = comm.allreduce(B_rotor_sum_local, op=MPI.SUM)
        
        if n_magnet_global > 0:
            B_magnet_avg = B_magnet_sum_global / n_magnet_global
        if n_rotor_global > 0:
            B_rotor_avg = B_rotor_sum_global / n_rotor_global
        
        # Airgap radius info (for debugging)
        if len(airgap_radii_local) > 0:
            r_min_local = float(airgap_radii_local.min())
            r_max_local = float(airgap_radii_local.max())
            r_min_global = comm.allreduce(r_min_local, op=MPI.MIN)
            r_max_global = comm.allreduce(r_max_local, op=MPI.MAX)
            airgap_radii_info = (r_min_global, r_max_global)
    
    return {
        'B_avg': B_avg_global,
        'B_max': B_max_global,
        'B_min': B_min_global,
        'B_rms': B_rms_global,
        'B_avg_vector': B_avg_vector_global,
        'n_cells': n_cells_global,
        'B_radial_avg': B_radial_avg_global,
        'B_tangential_avg': B_tangential_avg_global,
        'B_magnet_avg': B_magnet_avg,  # For comparison
        'B_rotor_avg': B_rotor_avg,    # For comparison
        'airgap_radii': airgap_radii_info,  # (min, max) radii of airgap cells
    }


def compute_torque_maxwell_surface(mesh, A_sol, facet_tags, config, comm=None, midair_tag: int = 2):
    """
    Compute electromagnetic torque about the z-axis using the Maxwell stress tensor
    integrated over a cylindrical surface in the airgap (MidAir).

    This is the standard approach:
      f = T · n,  where T = (1/μ0) (B ⊗ B - 0.5 |B|^2 I)
      τ_z = ∫ (x f_y - y f_x) dS   over MidAir surface

    Notes:
    - Requires facet tags to include the MidAir surface (tag=2 by default).
    - Uses B = curl(A) directly (no DG0/CG projection needed for the integral).
    """
    if comm is None:
        comm = mesh.comm

    if facet_tags is None:
        if comm.rank == 0:
            print("Warning: facet_tags is None; cannot compute Maxwell-surface torque.")
        return 0.0

    mu0 = float(config.mu0)
    B = ufl.curl(A_sol)
    n = ufl.FacetNormal(mesh)
    x = ufl.SpatialCoordinate(mesh)

    # Maxwell traction f = T·n
    Bn = ufl.dot(B, n)
    B2 = ufl.dot(B, B)
    f = (1.0 / mu0) * (Bn * B - 0.5 * B2 * n)

    # Torque density around z: x*f_y - y*f_x
    tau_z_form = (x[0] * f[1] - x[1] * f[0]) * ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)(midair_tag)

    torque_local = fem.assemble_scalar(fem.form(tau_z_form))
    torque_global = comm.allreduce(torque_local, op=MPI.SUM)
    return float(torque_global)

