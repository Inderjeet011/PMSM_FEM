"""3D A-V solver helpers: forms, assembly, and PETSc solver setup."""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl


def build_forms_submesh(
    mesh_parent,
    A_space,
    V_space,
    sigma,
    nu,
    J_z,
    M_vec,
    A_prev,
    dx_parent,
    dx_rs,
    dx_rpm,
    dx_c,
    dx_pm,
    config,
    entity_map,
    dx_cond_parent,
):
    """Build mixed A-V forms with A on parent mesh and V on conductor submesh."""
    mu0 = config.mu0

    A = ufl.TrialFunction(A_space)
    v = ufl.TestFunction(A_space)
    S = ufl.TrialFunction(V_space)
    q = ufl.TestFunction(V_space)

    curlA = ufl.curl(A)
    curlv = ufl.curl(v)
    xcoord = ufl.SpatialCoordinate(mesh_parent)
    omega = fem.Constant(mesh_parent, PETSc.ScalarType(config.omega_m))
    u_rot = ufl.cross(ufl.as_vector((0.0, 0.0, omega)), xcoord)
    inv_dt = fem.Constant(mesh_parent, PETSc.ScalarType(1.0 / config.dt))

    a00 = (
        nu * ufl.inner(curlA, curlv) * dx_parent
        + (sigma * inv_dt) * ufl.inner(A, v) * dx_rs
        + (sigma * inv_dt) * ufl.inner(A, v) * dx_c
        + (sigma * inv_dt) * ufl.inner(A, v) * dx_pm
        - sigma * ufl.inner(ufl.cross(u_rot, curlA), v) * dx_rpm
    )

    dx_rotor_iron = dx_parent(5)
    a01 = sigma * ufl.inner(ufl.grad(S), v) * dx_cond_parent
    a10 = (
        -(sigma * inv_dt) * ufl.inner(A, ufl.grad(q)) * dx_cond_parent
        + sigma * ufl.inner(ufl.cross(u_rot, curlA), ufl.grad(q)) * dx_rotor_iron
    )
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


def assemble_system_matrix_submesh(
    mesh_parent,
    block_bcs,
    A_space_parent,
    V_space_submesh,
    a_block_form,
):
    """Assemble monolithic mixed matrix then split into nested sub-blocks."""
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


def configure_solver_submesh(
    mesh_parent,
    mat_nest,
    mat_blocks,
    A_space,
    V_space,
    config,
):
    """Configure the mixed solver used by main_submesh.py."""
    comm = mesh_parent.comm
    A00_full = mat_blocks[0][0]

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(mat_nest, mat_nest)
    ksp.setType("fgmres")
    ksp.setGMRESRestart(150)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setTolerances(
        rtol=0.0,
        atol=float(config.outer_atol),
        max_it=int(config.outer_max_it),
    )

    def _outer_ksp_monitor(ksp_obj, its, rnorm):
        if comm.rank == 0:
            print(f"  [outer KSP] it={its:3d}  |r|={rnorm:.6e}")

    ksp.setMonitor(_outer_ksp_monitor)
    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.LOWER)
    pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.A11)
    isA, isV = mat_nest.getNestISs()
    pc.setFieldSplitIS(("A", isA[0]), ("V", isV[1]))
    pc.setUp()

    ksp_A, ksp_V = pc.getFieldSplitSubKSP()
    ksp_A.setType("fgmres")
    ksp_A.setGMRESRestart(int(config.ksp_A_restart))
    ksp_A.setTolerances(
        rtol=float(getattr(config, "ksp_A_rtol", 1e-5)),
        atol=float(getattr(config, "ksp_A_atol", 1e-12)),
        max_it=int(config.ksp_A_max_it),
    )
    ksp_A.setOperators(A00_full, A00_full)

    pc_A = ksp_A.getPC()
    pc_A.setType("hypre")
    pc_A.setHYPREType("boomeramg")

    opts = PETSc.Options()
    prefix_A = ksp_A.getOptionsPrefix() or ""
    opts[f"{prefix_A}pc_hypre_boomeramg_max_iter"] = 20
    opts[f"{prefix_A}pc_hypre_boomeramg_tol"] = 0.0
    opts[f"{prefix_A}pc_hypre_boomeramg_coarsen_type"] = "HMIS"
    ksp_A.setFromOptions()

    ksp_V.setType("preonly")
    pc_V = ksp_V.getPC()
    pc_V.setType("hypre")
    pc_V.setHYPREType("boomeramg")
    opts = PETSc.Options()
    prefix_V = ksp_V.getOptionsPrefix() or ""
    opts[f"{prefix_V}pc_hypre_boomeramg_max_iter"] = 10
    opts[f"{prefix_V}pc_hypre_boomeramg_tol"] = 0.0
    opts[f"{prefix_V}pc_hypre_boomeramg_coarsen_type"] = "HMIS"
    ksp_V.setFromOptions()

    ksp.setFromOptions()
    return ksp




 