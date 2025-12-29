"""3D A-V solver helpers: forms, assembly, and PETSc solver setup."""

from dolfinx import fem
from dolfinx.fem import petsc
from dolfinx.cpp.fem.petsc import discrete_gradient
from petsc4py import PETSc
import ufl

_keepalive = []





def build_forms(mesh, A_space, V_space, sigma, nu, J_z, M_vec, A_prev,
                dx, dx_cond_stat, dx_cond_rot, dx_coils, dx_magnets, ds, config, exterior_facet_tag=None):
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
    
    inv_dt = fem.Constant(mesh, PETSc.ScalarType(1.0 / config.dt))

    # A-equation (eddy-current A–V formulation):
    #   ∫ nu curl(A)·curl(v)  +  ∫_{cond} (sigma/dt) A^{n+1}·v  +  ∫_{cond} sigma ∇V^{n+1}·v  = RHS
    # Fixes:
    # - remove spurious μ0 multiplying σ-terms
    # - restrict all σ-terms to dx_cond_stat / dx_cond_rot
    a00 = (
        nu * ufl.inner(curlA, curlv) * dx
        + (sigma * inv_dt) * ufl.inner(A, v) * dx_cond_stat
        - sigma * ufl.inner(ufl.cross(u_rot, curlA), v) * dx_cond_rot
    )
    epsA_full = fem.Constant(mesh, PETSc.ScalarType(float(getattr(config, "epsilon_A_full", 0.0))))
    a00 += epsA_full * ufl.inner(A, v) * dx
    
    epsilon_A_spd = float(getattr(config, "epsilon_A_spd", 1e-6))
    epsilon = fem.Constant(mesh, PETSc.ScalarType(epsilon_A_spd))
    # AMS-friendly SPD approximation of (dt * a00): dt*nu*curlcurl + sigma*mass (conductors only) + regularization
    dx_cond_all = dx_cond_stat + dx_cond_rot
    a00_spd = (
        dt * nu * ufl.inner(curlA, curlv) * dx
        + sigma * ufl.inner(A, v) * dx_cond_all
        + epsilon * ufl.inner(A, v) * dx
    )
    
    # A–V coupling terms (conductors only, no μ0):
    # - a01: ∫_{cond} sigma ∇V^{n+1}·v
    # - a10/a11: V-equation ∫_{cond} sigma ∇V^{n+1}·∇q + ∫_{cond} (sigma/dt) A^{n+1}·∇q = RHS(A^n)
    a01 = sigma * ufl.inner(ufl.grad(S), v) * dx_cond_stat
    a10 = (
        -(sigma * inv_dt) * ufl.inner(A, ufl.grad(q)) * dx_cond_stat
        + sigma * ufl.inner(ufl.cross(u_rot, curlA), ufl.grad(q)) * dx_cond_rot
    )
    a11_core = sigma * ufl.inner(ufl.grad(S), ufl.grad(q)) * dx_cond_stat
    epsV = fem.Constant(mesh, PETSc.ScalarType(1e-8 * mu0))
    a11 = a11_core + epsV * ufl.inner(S, q) * dx_cond_stat

    # IMPORTANT: do NOT scale (a10, a11) by dt here — that would cancel the time derivative
    # in the V-equation and break the standard eddy-current A–V formulation.
    
    #Right hand side
    # Impressed current density is applied only in the coil subdomains.
    J_term = J_z * v[2] * dx_coils
    # Fix: σ-term uses (sigma/dt) A^n and is restricted to conductors; remove μ0.
    lagging_A = (sigma * inv_dt) * ufl.inner(A_prev, v) * dx_cond_stat
    # PM source term (standard): uses rotating M_vec(t) updated each timestep.
    pm_term = ufl.inner(nu * mu0 * M_vec, curlv) * dx_magnets
    L0 = J_term + lagging_A + pm_term
    
    zero_scalar = fem.Constant(mesh, PETSc.ScalarType(0))
    # V-equation RHS: move -(sigma/dt) A^n·∇q to the RHS (do not cancel the time derivative).
    L1 = (sigma * inv_dt) * ufl.inner(A_prev, ufl.grad(q)) * dx_cond_stat + zero_scalar * q * dx
    
    a_blocks = ((fem.form(a00), fem.form(a01)), (fem.form(a10), fem.form(a11)))
    a00_spd_form = fem.form(a00_spd)
    # No motional EMF term in this (original) formulation; keep return value for compatibility.
    zero_scalar = fem.Constant(mesh, PETSc.ScalarType(0))
    a00_motional_form = fem.form(zero_scalar * v[0] * dx)
    L_blocks = (fem.form(L0), fem.form(L1))

    return a_blocks, L_blocks, a00_spd_form, a00_motional_form





def assemble_system_matrix(mesh, a_blocks, block_bcs, a00_spd_form=None, a00_motional_form=None, beta_pc=0.3):
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
    
    A00_spd = None
    if a00_spd_form is not None:
        A00_spd = petsc.assemble_matrix(a00_spd_form, bcs=None)
        A00_spd.assemble()
        A00_spd.setOption(PETSc.Mat.Option.SPD, True)

    return mats, mat_nest, A00_standalone, A00_spd, None



def configure_solver(mesh, mat_nest, mat_blocks, A_space, V_space, A00_spd, config):
    A00_full = mat_blocks[0][0]

    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(mat_nest, mat_nest)
    ksp.setType("fgmres")
    ksp.setTolerances(
        rtol=float(getattr(config, "outer_rtol", 1e-4)),
        atol=0.0,
        max_it=int(getattr(config, "outer_max_it", 50)),
    )

    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.LOWER)
    pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.A11)

    isA, isV = mat_nest.getNestISs()
    pc.setFieldSplitIS(("A", isA[0]), ("V", isV[1]))
    pc.setUp()
    ksp_A, ksp_V = pc.getFieldSplitSubKSP()

    # A-block AMS
    V_ams = fem.functionspace(mesh, ("Lagrange", 1))
    G = discrete_gradient(V_ams._cpp_object, A_space._cpp_object)
    G.assemble()

    xcoord = ufl.SpatialCoordinate(mesh)
    verts = []
    for dim in range(3):
        f = fem.Function(V_ams)
        f.interpolate(fem.Expression(xcoord[dim], V_ams.element.interpolation_points))
        f.x.scatter_forward()
        verts.append(f.x.petsc_vec)

    e0 = G.createVecLeft()
    e1 = G.createVecLeft()
    e2 = G.createVecLeft()
    G.mult(verts[0], e0)
    G.mult(verts[1], e1)
    G.mult(verts[2], e2)

    ksp_A.setType("fgmres")
    ksp_A.setTolerances(rtol=0.0, atol=0.0, max_it=int(getattr(config, "ksp_A_max_it", 2)))
    ksp_A.setGMRESRestart(int(getattr(config, "ksp_A_restart", 30)))
    ksp_A.setOperators(A00_full, A00_spd if A00_spd is not None else A00_full)

    pc_A = ksp_A.getPC()
    pc_A.setType("hypre")
    pc_A.setHYPREType("ams")
    pc_A.setHYPREDiscreteGradient(G)
    pc_A.setHYPRESetEdgeConstantVectors(e0, e1, e2)

    ksp_V.setType("preonly")
    pc_V = ksp_V.getPC()
    pc_V.setType("hypre")
    pc_V.setHYPREType("boomeramg")

    _keepalive.append((G, V_ams, (e0, e1, e2)))
    return ksp




 