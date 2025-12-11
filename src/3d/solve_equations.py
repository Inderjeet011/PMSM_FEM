"""Solver functions: forms, assembly, sources."""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl

from load_mesh import CURRENT_MAP, MAGNETS


def setup_sources(mesh, cell_tags):
    """Create current and magnetization fields."""
    DG0 = fem.functionspace(mesh, ("DG", 0))
    DG0_vec = fem.functionspace(mesh, ("DG", 0, (3,)))
    J_z = fem.Function(DG0, name="Jz")
    M_vec = fem.Function(DG0_vec, name="M")
    return J_z, M_vec


def initialise_magnetisation(mesh, cell_tags, M_vec, config):
    """Initialize permanent magnet magnetization."""
    dofmap = mesh.geometry.dofmap
    coords = mesh.geometry.x
    vec_view = M_vec.x.array.reshape((-1, 3))
    magnitude = config.magnet_remanence / max(config.mu0, 1e-12)
    
    for marker in MAGNETS:
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
        print("Magnetization initialized")


def rotate_magnetization(mesh, cell_tags, M_vec, config, t):
    """Rotate magnetization with rotor."""
    theta_rot = config.omega_m * t
    dofmap = mesh.geometry.dofmap
    coords = mesh.geometry.x
    vec_view = M_vec.x.array.reshape((-1, 3))
    magnitude = config.magnet_remanence / max(config.mu0, 1e-12)
    
    pm_spacing = (np.pi / 6) + (np.pi / 30)
    pm_angles = np.asarray([i * pm_spacing for i in range(10)])
    
    def get_pm_sign(marker):
        idx = marker - 13
        return 1 if (idx % 2 == 0) else -1
    
    for marker in MAGNETS:
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
    """Update coil currents."""
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
    """Get max current density."""
    if J_z is None:
        return 0.0
    return float(np.max(np.abs(J_z.x.array)))


def build_forms(mesh, A_space, V_space, sigma, nu, J_z, M_vec, A_prev,
                dx, dx_conductors, dx_magnets, config):
    """Build variational forms."""
    dt = fem.Constant(mesh, PETSc.ScalarType(config.dt))
    mu0 = config.mu0
    xcoord = ufl.SpatialCoordinate(mesh)
    omega = fem.Constant(mesh, PETSc.ScalarType(config.omega_m))
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
    gauge = fem.Constant(mesh, PETSc.ScalarType(1e-10))
    a10 = gauge * ufl.div(A) * q * dx
    a11 = mu0 * sigma * ufl.inner(ufl.grad(S), ufl.grad(q)) * dx
    
    J_term = dt * mu0 * J_z * v[2] * dx
    lagging = sigma * mu0 * ufl.inner(A_prev, v) * dx
    pm_term = -ufl.inner(M_vec, curlv) * dx_magnets
    L0 = J_term + lagging + pm_term
    
    zero_scalar = fem.Constant(mesh, PETSc.ScalarType(0))
    L1 = zero_scalar * q * dx
    
    a_blocks = ((fem.form(a00), fem.form(a01)), (fem.form(a10), fem.form(a11)))
    L_blocks = (fem.form(L0), fem.form(L1))
    
    if mesh.comm.rank == 0:
        print("Forms built")
    
    return a_blocks, L_blocks


def rebuild_linear_forms(mesh, A_space, V_space, sigma, J_z, M_vec, A_prev,
                         dx, dx_magnets, config):
    """Rebuild linear forms when sources change."""
    mu0 = config.mu0
    dt = fem.Constant(mesh, PETSc.ScalarType(config.dt))
    
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
    """Assemble system matrix."""
    mats = [[None, None], [None, None]]
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
    mat_nest = PETSc.Mat().createNest(mats, comm=mesh.comm)
    mat_nest.assemble()
    
    if mesh.comm.rank == 0:
        print("Matrix assembled")
    
    return mats, mat_nest, A00_standalone


def configure_solver(mesh, mat_nest, mat_blocks, A_space, V_space, degree_A=None):
    """Configure linear solver."""
    A = mat_nest
    A00 = mat_blocks[0][0]
    A11 = mat_blocks[1][1]
    A11.setOption(PETSc.Mat.Option.SPD, True)
    
    P = PETSc.Mat().createNest([[A00, None], [None, A11]], comm=mesh.comm)
    P.assemble()
    
    ksp = PETSc.KSP().create(comm=mesh.comm)
    ksp.setOperators(A, P)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=1e-4, atol=1e-8, max_it=100)
    
    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    
    nested_IS = P.getNestISs()
    pc.setFieldSplitIS(("A", nested_IS[0][0]), ("V", nested_IS[1][1]))
    
    ksp_A, ksp_V = pc.getFieldSplitSubKSP()
    
    ksp_A.setType("preonly")
    pc_A = ksp_A.getPC()
    pc_A.setType("gamg")
    pc_A.setFromOptions()
    
    ksp_V.setType("preonly")
    pc_V = ksp_V.getPC()
    pc_V.setType("gamg")
    pc_V.setFromOptions()
    
    if mesh.comm.rank == 0:
        print("Solver configured")
    
    return ksp
