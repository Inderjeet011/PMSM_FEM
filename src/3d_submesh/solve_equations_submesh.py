"""3D A-V solver with submesh: forms, assembly, and PETSc solver setup.
A lives on parent mesh, V lives on conductor submesh.
"""

from dolfinx import fem  # type: ignore
from dolfinx.fem import petsc  # type: ignore
from petsc4py import PETSc  # type: ignore
import ufl  # type: ignore
import numpy as np  # type: ignore


def build_forms_submesh(mesh_parent, A_space, V_space,
                        sigma, nu, J_z, M_vec, A_prev,
                        dx_parent, dx_rs, dx_rpm, dx_c, dx_pm,
                        config, entity_map, dx_cond_parent):
    dt = fem.Constant(mesh_parent, PETSc.ScalarType(config.dt))
    mu0 = config.mu0

    A = ufl.TrialFunction(A_space)
    v = ufl.TestFunction(A_space)
    S = ufl.TrialFunction(V_space)
    q = ufl.TestFunction(V_space)
    
    curlA = ufl.curl(A)
    curlv = ufl.curl(v)
    x_coord = ufl.SpatialCoordinate(mesh_parent)
    omega_m = fem.Constant(mesh_parent, PETSc.ScalarType(config.omega_m))
    u_rot = ufl.as_vector((-omega_m * x_coord[1], omega_m * x_coord[0], 0.0))
    
    inv_dt = fem.Constant(mesh_parent, PETSc.ScalarType(1.0 / config.dt))
    
    # A-block: (σ/dt) in dx_rs (rotor, stator, aluminium) + dx_c + dx_pm (coils, magnets) for iterative convergence.
    # Avoid double-counting rotor: dx_rs already has rotor; dx_c + dx_pm add only coils and magnets.
    a00 = (
        nu * ufl.inner(curlA, curlv) * dx_parent
        + (sigma * inv_dt) * ufl.inner(A, v) * dx_rs
        + (sigma * inv_dt) * ufl.inner(A, v) * dx_c
        + (sigma * inv_dt) * ufl.inner(A, v) * dx_pm
        - sigma * ufl.inner(ufl.cross(u_rot, curlA), v) * dx_rpm
    )

    # A–V coupling (match 3d: σ, σ/dt). Rotation term on dx_rpm causes IndexMap error with submesh entity_maps.
    a01 = sigma * ufl.inner(ufl.grad(S), v) * dx_cond_parent
    a10 = -(sigma * inv_dt) * ufl.inner(A, ufl.grad(q)) * dx_cond_parent
    a11 = sigma * ufl.inner(ufl.grad(S), ufl.grad(q)) * dx_cond_parent

    L0 = (
        J_z * v[2] * dx_c
        + (sigma * inv_dt) * ufl.inner(A_prev, v) * dx_rs
        + (sigma * inv_dt) * ufl.inner(A_prev, v) * dx_c
        + (sigma * inv_dt) * ufl.inner(A_prev, v) * dx_pm
        + ufl.inner(nu * mu0 * M_vec, curlv) * dx_pm
    )
    L1 = (sigma * inv_dt) * ufl.inner(A_prev, ufl.grad(q)) * dx_cond_parent

    em = [entity_map]
    a_blocks = (
        (fem.form(a00), fem.form(a01, entity_maps=em)),
        (fem.form(a10, entity_maps=em), fem.form(a11, entity_maps=em)),
    )
    L_blocks = (fem.form(L0), fem.form(L1, entity_maps=em))
    a_block_form = fem.form([[a00, a01], [a10, a11]], entity_maps=em)
    return a_blocks, L_blocks, a_block_form


def assemble_system_matrix_submesh(mesh_parent, a_blocks, block_bcs,
                                    A_space_parent, V_space_submesh, a_block_form):
    comm = mesh_parent.comm
    n_A_dofs = A_space_parent.dofmap.index_map.size_global
    n_V_dofs = V_space_submesh.dofmap.index_map.size_global

    bcs_flat = [bc for bclist in block_bcs for bc in bclist]
    A_mono = petsc.assemble_matrix(a_block_form, bcs=bcs_flat)
    A_mono.assemble()
    is_A = PETSc.IS().createGeneral(np.arange(0, n_A_dofs, dtype=np.int32), comm=comm)
    is_V = PETSc.IS().createGeneral(
        np.arange(n_A_dofs, n_A_dofs + n_V_dofs, dtype=np.int32), comm=comm
    )
    mats = [
        [A_mono.createSubMatrix(is_A, is_A), A_mono.createSubMatrix(is_A, is_V)],
        [A_mono.createSubMatrix(is_V, is_A), A_mono.createSubMatrix(is_V, is_V)],
    ]
    is_A.destroy()
    is_V.destroy()

    mat_nest = PETSc.Mat().createNest(mats, comm=comm)
    mat_nest.assemble()

    return mats, mat_nest


def configure_solver_submesh(mesh_parent, mat_nest, mat_blocks, A_space, V_space, config):
    comm = mesh_parent.comm
    A00_full = mat_blocks[0][0]

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(mat_nest, mat_nest)
    ksp.setType("fgmres")
    ksp.setGMRESRestart(150)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setTolerances(
        rtol=0.0,  # disable relative tol so solver runs until atol or max_it
        atol=float(config.outer_atol),
        max_it=int(config.outer_max_it),
    )
    def _outer_ksp_monitor(ksp_obj, its, rnorm):
        if comm.rank == 0:
            print(f"  [outer KSP] it={its:3d}  |r|={rnorm:.6e}")

    ksp.setMonitor(_outer_ksp_monitor)
    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    isA, isV = mat_nest.getNestISs()
    pc.setFieldSplitIS(("A", isA[0]), ("V", isV[1]))
    pc.setUp()

    ksp_A, ksp_V = pc.getFieldSplitSubKSP()

    ksp_A.setType("fgmres")
    ksp_A.setGMRESRestart(int(config.ksp_A_restart))
    ksp_A.setTolerances(
        max_it=int(config.ksp_A_max_it),
    )
    ksp_A.setOperators(A00_full, A00_full)

    pc_A = ksp_A.getPC()
    pc_A.setType("hypre")
    pc_A.setHYPREType("boomeramg")
    opts = PETSc.Options()
    opts[f"{ksp_A.prefix}pc_hypre_boomeramg_max_iter"] = 20
    opts[f"{ksp_A.prefix}pc_hypre_boomeramg_tol"] = 0.0
    opts[f"{ksp_A.prefix}pc_hypre_boomeramg_coarsen_type"] = "HMIS"
    ksp_A.setFromOptions()

    ksp_V.setType("preonly")
    pc_V = ksp_V.getPC()
    pc_V.setType("hypre")
    pc_V.setHYPREType("boomeramg")
    opts = PETSc.Options()
    opts[f"{ksp_V.prefix}pc_hypre_boomeramg_max_iter"] = 20
    opts[f"{ksp_V.prefix}pc_hypre_boomeramg_tol"] = 0.0
    opts[f"{ksp_V.prefix}pc_hypre_boomeramg_coarsen_type"] = "HMIS"
    ksp_V.setFromOptions()

    ksp.setFromOptions()
    return ksp
