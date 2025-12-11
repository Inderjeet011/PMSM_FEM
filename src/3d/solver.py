"""
Solver module for 3D A-V solver.

This module handles:
- Form building and assembly
- Solver configuration
- Source management (currents and magnetization)
"""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl

from setup import CURRENT_MAP, DomainTags3D


# ============================================================================
# Source management
# ============================================================================

def setup_sources(mesh, cell_tags):
    """Allocate coil current density & magnetisation fields."""
    DG0 = fem.functionspace(mesh, ("DG", 0))
    DG0_vec = fem.functionspace(mesh, ("DG", 0, (3,)))

    J_z = fem.Function(DG0, name="Jz")
    M_vec = fem.Function(DG0_vec, name="M")

    return J_z, M_vec


def initialise_magnetisation(mesh, cell_tags, M_vec, config):
    """Place-holder PM magnetisation: radial in xy-plane."""
    dofmap = mesh.geometry.dofmap
    coords = mesh.geometry.x
    vec_view = M_vec.x.array.reshape((-1, 3))

    magnitude = config.magnet_remanence / max(config.mu0, 1e-12)
    
    if mesh.comm.rank == 0:
        print(f"🧲 Magnetisation: Br = {config.magnet_remanence:.3f} T")
        print(f"   M magnitude = {magnitude:.3e} A/m")
        print(f"   Expected B ≈ {config.magnet_remanence:.3f} T in magnets")

    for marker in DomainTags3D.MAGNETS:
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        for c in cells:
            geom_dofs = dofmap[c]
            cell_coords = coords[geom_dofs]
            cx = float(np.mean(cell_coords[:, 0]))
            cy = float(np.mean(cell_coords[:, 1]))
            norm = np.hypot(cx, cy)
            if norm < 1e-12:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = np.array([cx / norm, cy / norm, 0.0])
            vec_view[c, :] = magnitude * direction

    if mesh.comm.rank == 0:
        sample_cells = cell_tags.find(DomainTags3D.MAGNETS[0])
        if sample_cells.size > 0:
            sample_M = vec_view[sample_cells[0], :]
            print(f"   Sample M in magnet: {sample_M}")
            print(f"   |M| = {np.linalg.norm(sample_M):.3e} A/m")
        print("🧲 Magnetisation initialised (radial placeholder)")


def rotate_magnetization(mesh, cell_tags, M_vec, config, t):
    """Rotate PM magnetization with rotor (3D version)."""
    theta_rot = config.omega_m * t
    dofmap = mesh.geometry.dofmap
    coords = mesh.geometry.x
    vec_view = M_vec.x.array.reshape((-1, 3))
    
    magnitude = config.magnet_remanence / max(config.mu0, 1e-12)
    
    pm_spacing = (np.pi / 6) + (np.pi / 30)
    pm_angles = np.asarray([i * pm_spacing for i in range(10)])
    
    def get_pm_sign(marker: int) -> int:
        """Return +1 for N pole, -1 for S pole"""
        idx = marker - 13
        return 1 if (idx % 2 == 0) else -1
    
    for marker in DomainTags3D.MAGNETS:
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        
        sign = get_pm_sign(marker)
        pm_idx = marker - 13
        theta_pole_center = pm_angles[pm_idx]
        
        for c in cells:
            geom_dofs = dofmap[c]
            cell_coords = coords[geom_dofs]
            cx = float(np.mean(cell_coords[:, 0]))
            cy = float(np.mean(cell_coords[:, 1]))
            
            theta = np.arctan2(cy, cx)
            if theta < 0:
                theta += 2 * np.pi
            
            theta_now = theta_pole_center + theta_rot
            
            vec_view[c, 0] = sign * magnitude * np.cos(theta_now)
            vec_view[c, 1] = sign * magnitude * np.sin(theta_now)
            vec_view[c, 2] = 0.0
    
    M_vec.x.scatter_forward()


def update_currents(mesh, cell_tags, J_z, config, t):
    """Populate J_z for each coil using three-phase mapping."""
    omega = config.omega_e
    J_peak = config.coil_current_peak
    J_z.x.array[:] = 0.0
    for marker, meta in CURRENT_MAP.items():
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        drive = meta["alpha"] * np.sin(omega * t + meta["beta"])
        J_z.x.array[cells] = J_peak * drive


def current_stats(J_z):
    """Return max |J_z| for diagnostics."""
    if J_z is None:
        return 0.0
    return float(np.max(np.abs(J_z.x.array)))


# ============================================================================
# Form building and assembly
# ============================================================================

def build_forms(mesh, A_space, V_space, sigma, nu, J_z, M_vec, A_prev, 
                dx, dx_conductors, dx_magnets, config):
    """Create block bilinear/linear forms."""
    dt = fem.Constant(mesh, PETSc.ScalarType(config.dt))  # type: ignore
    mu0 = config.mu0
    xcoord = ufl.SpatialCoordinate(mesh)
    omega = fem.Constant(mesh, PETSc.ScalarType(config.omega_m))  # type: ignore
    u_rot = ufl.as_vector((-omega * xcoord[1], omega * xcoord[0], 0.0))

    A = ufl.TrialFunction(A_space)
    v = ufl.TestFunction(A_space)
    S = ufl.TrialFunction(V_space)
    q = ufl.TestFunction(V_space)

    curlA = ufl.curl(A)
    curlv = ufl.curl(v)

    a00 = dt * nu * ufl.inner(curlA, curlv) * dx
    a00 += sigma * mu0 * ufl.inner(A, v) * dx
    a00 += sigma * mu0 * ufl.inner(ufl.cross(u_rot, curlA), v) * dx_conductors

    a01 = mu0 * sigma * ufl.inner(v, ufl.grad(S)) * dx

    gauge = fem.Constant(mesh, PETSc.ScalarType(1e-10))  # type: ignore
    a10 = gauge * ufl.div(A) * q * dx

    a11 = mu0 * sigma * ufl.inner(ufl.grad(S), ufl.grad(q)) * dx

    A_prev.name = "A_prev"

    J_term = dt * mu0 * J_z * v[2] * dx
    lagging = sigma * mu0 * ufl.inner(A_prev, v) * dx
    pm_term = -ufl.inner(M_vec, curlv) * dx_magnets
    L0 = J_term + lagging + pm_term

    zero_scalar = fem.Constant(mesh, PETSc.ScalarType(0))
    L1 = zero_scalar * q * dx

    a_blocks = (
        (fem.form(a00), fem.form(a01)),
        (fem.form(a10), fem.form(a11)),
    )
    L_blocks = (fem.form(L0), fem.form(L1))
    
    if mesh.comm.rank == 0:
        print("🧮 Variational forms assembled (motional term enabled).")

    return a_blocks, L_blocks


def rebuild_linear_forms(mesh, A_space, V_space, sigma, J_z, M_vec, A_prev,
                         dx, dx_magnets, config):
    """Rebuild only linear forms when M_vec or J_z changes."""
    mu0 = config.mu0
    dt = fem.Constant(mesh, PETSc.ScalarType(config.dt))  # type: ignore
    
    v = ufl.TestFunction(A_space)
    q = ufl.TestFunction(V_space)
    curlv = ufl.curl(v)
    
    J_term = dt * mu0 * J_z * v[2] * dx
    lagging = sigma * mu0 * ufl.inner(A_prev, v) * dx
    pm_term = -ufl.inner(M_vec, curlv) * dx_magnets
    L0 = J_term + lagging + pm_term
    
    zero_scalar = fem.Constant(mesh, PETSc.ScalarType(0))
    L1 = zero_scalar * q * dx
    
    return (fem.form(L0), fem.form(L1))


def assemble_system_matrix(mesh, a_blocks, block_bcs):
    """Create (and keep) the nested PETSc operator."""
    if a_blocks is None:
        raise RuntimeError("Forms not built.")

    mats: list[list[PETSc.Mat]] = [[None, None], [None, None]]  # type: ignore
    for i in range(2):
        for j in range(2):
            bcs_for_block = []
            if block_bcs[i]:
                bcs_for_block.extend(block_bcs[i])
            mat = petsc.assemble_matrix(a_blocks[i][j], bcs=bcs_for_block if bcs_for_block else None)
            mat.assemble()
            mats[i][j] = mat

    A00_standalone = mats[0][0].copy()
    A00_standalone.assemble()
    
    mat_nest = PETSc.Mat().createNest(mats, comm=mesh.comm)  # type: ignore[arg-type]
    mat_nest.assemble()
    
    if mesh.comm.rank == 0:
        print("🧱 System matrix assembled (nest).")
    
    return mats, mat_nest, A00_standalone


# ============================================================================
# Solver configuration
# ============================================================================

def configure_solver(mesh, mat_nest, mat_blocks, A_space, V_space, degree_A=None):
    """Configure KSP with GMRES and FieldSplit preconditioner.
    
    Preconditioner strategy:
    - A block (H(curl)): AMS (Auxiliary-space Maxwell Solver) - designed for curl-curl problems
    - V block (H1): GAMG (Geometric Algebraic Multigrid) - standard for H1/Laplacian problems
    """
    if mat_nest is None:
        raise RuntimeError("Matrix not assembled. Call setup() first.")

    A = mat_nest
    A00 = mat_blocks[0][0]
    A11 = mat_blocks[1][1]
    A11.setOption(PETSc.Mat.Option.SPD, True)

    P = PETSc.Mat().createNest([[A00, None],
                                [None, A11]],
                                comm=mesh.comm)
    P.assemble()

    ksp = PETSc.KSP().create(comm=mesh.comm)  # type: ignore
    ksp.setOperators(A, P)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=1e-4, atol=1e-8, max_it=100)
    ksp.setConvergenceHistory()

    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    nested_IS = P.getNestISs()
    pc.setFieldSplitIS(
        ("A", nested_IS[0][0]),
        ("V", nested_IS[1][1])
    )

    ksp_A, ksp_V = pc.getFieldSplitSubKSP()

    # A block: Use GAMG for H(curl) space
    # Note: AMS would be ideal but requires complex setup (discrete gradient, 
    # interpolation matrix, coordinates). GAMG works well as a robust alternative.
    # TODO: Can upgrade to AMS later with proper configuration
    ksp_A.setType("preonly")
    pc_A = ksp_A.getPC()
    pc_A.setType("gamg")
    pc_A.setFromOptions()

    # V block: Use GAMG for H1/Lagrange space
    ksp_V.setType("preonly")
    pc_V = ksp_V.getPC()
    pc_V.setType("gamg")
    pc_V.setFromOptions()

    if mesh.comm.rank == 0:
        print("\n⚙️  Solver configured:")
        print("   Type: GMRES")
        print("   Preconditioner: FieldSplit")
        print("     - V-block: GAMG (Geometric Algebraic Multigrid) ✓")
        print("   Tolerances: rtol=1e-4, atol=1e-8, max_it=100")
        print()

    return ksp

