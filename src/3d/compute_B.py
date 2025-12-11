"""Post-processing: compute B field from A."""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl


def compute_B_field(mesh, A_sol, B_space, B_magnitude_space, config, cell_tags=None, debug=False):
    """Compute B = curl(A)."""
    if debug and mesh.comm.rank == 0:
        print("Computing B = curl(A)...")
    
    B_sol = fem.Function(B_space, name="B")
    curlA = ufl.curl(A_sol)
    
    B_test = ufl.TestFunction(B_space)
    B_trial = ufl.TrialFunction(B_space)
    a_B = fem.form(ufl.inner(B_trial, B_test) * ufl.dx)
    L_B = fem.form(ufl.inner(curlA, B_test) * ufl.dx)
    
    A_B = petsc.assemble_matrix(a_B)
    A_B.assemble()
    b_B = petsc.create_vector(L_B)
    petsc.assemble_vector(b_B, L_B)
    b_B.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    ksp_B = PETSc.KSP().create(comm=mesh.comm)
    ksp_B.setOperators(A_B)
    ksp_B.setType("cg")
    pc_B = ksp_B.getPC()
    pc_B.setType("hypre")
    try:
        pc_B.setHYPREType("boomeramg")
    except:
        pc_B.setType("jacobi")
    
    ksp_B.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)
    x_B = b_B.duplicate()
    ksp_B.solve(b_B, x_B)
    B_sol.x.array[:] = x_B.array_r[:B_sol.x.array.size]
    B_sol.x.scatter_forward()
    
    A_B.destroy()
    b_B.destroy()
    x_B.destroy()
    ksp_B.destroy()
    
    B_sol.name = "B"
    
    # Compute magnitude
    B_array = B_sol.x.array.reshape((-1, 3))
    B_magnitude = np.linalg.norm(B_array, axis=1)
    
    B_magnitude_sol = fem.Function(B_magnitude_space, name="B_Magnitude")
    B_magnitude_sol.x.array[:] = B_magnitude
    B_magnitude_sol.x.scatter_forward()
    
    # No clipping - use raw computed values
    max_B = float(np.max(B_magnitude))
    min_B = float(np.min(B_magnitude))
    norm_B = float(np.linalg.norm(B_array))
    
    if debug and mesh.comm.rank == 0:
        print(f"B field: max={max_B:.3e} T, min={min_B:.3e} T, ||B||={norm_B:.3e}")
    
    return B_sol, B_magnitude_sol, max_B, min_B, norm_B
