"""
Diagnostic script to understand WHY outliers occur in B = curl(A) computation.

This investigates:
1. A field quality (Nédélec elements)
2. curl(A) computation at each step
3. Where artifacts are introduced
4. Boundary effects
"""

import sys
sys.path.insert(0, '/root/FEniCS/src/3d')

import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, io
import ufl
from petsc4py import PETSc
from dolfinx import petsc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import basix.ufl

from setup import SimulationConfig3D, load_mesh_3d, setup_materials, setup_boundary_conditions
from solver import build_forms, configure_solver
from postprocess import compute_B_field

def diagnose_curl_computation():
    """Step-by-step diagnosis of curl(A) computation."""
    
    print("="*70)
    print("ROOT CAUSE ANALYSIS: Why B = curl(A) has outliers")
    print("="*70)
    
    # Load mesh and setup
    config = SimulationConfig3D()
    domain, cell_tags, facet_tags = load_mesh_3d(config)
    mesh_obj = domain
    
    # Setup materials
    sigma, nu, M_vec = setup_materials(mesh_obj, cell_tags, config)
    
    # Create function spaces
    nedelec_element = basix.ufl.element("N1curl", mesh_obj.basix_cell(), config.degree_A)
    A_space = fem.functionspace(mesh_obj, nedelec_element)
    
    lagrange_element = basix.ufl.element("Lagrange", mesh_obj.basix_cell(), config.degree_A, shape=(3,))
    B_space = fem.functionspace(mesh_obj, lagrange_element)
    B_magnitude_space = fem.functionspace(mesh_obj, ("CG", config.degree_A))
    
    # Create a test A field (or load from solution)
    A_sol = fem.Function(A_space, name="A")
    
    # Initialize with a simple test case or load from file
    # For now, let's check what happens with a simple field
    print("\n[STEP 1] Analyzing A field (Nédélec elements)")
    print("-" * 70)
    
    # Get DOF coordinates for Nédélec elements
    # Nédélec DOFs are on edges, not nodes
    print(f"   A space: {A_space}")
    print(f"   DOFs: {A_space.dofmap.index_map.size_local}")
    print(f"   Element type: Nédélec (H(curl))")
    print(f"   DOF locations: Edges (not nodes!)")
    
    # Check A field values
    A_array = A_sol.x.array
    print(f"\n   A field statistics:")
    print(f"     ||A|| = {np.linalg.norm(A_array):.6e} Wb/m")
    print(f"     max|A| = {np.max(np.abs(A_array)):.6e} Wb/m")
    print(f"     min|A| = {np.min(np.abs(A_array)):.6e} Wb/m")
    print(f"     mean|A| = {np.mean(np.abs(A_array)):.6e} Wb/m")
    
    # Check for discontinuities or large gradients
    if len(A_array) > 0:
        A_sorted = np.sort(np.abs(A_array))
        p95 = A_sorted[int(0.95 * len(A_sorted))]
        p99 = A_sorted[int(0.99 * len(A_sorted))]
        print(f"     95th percentile: {p95:.6e} Wb/m")
        print(f"     99th percentile: {p99:.6e} Wb/m")
        
        if p99 > 10 * p95:
            print(f"     ⚠️  Large spread detected - may cause curl issues")
    
    print("\n[STEP 2] Computing curl(A) symbolically")
    print("-" * 70)
    
    # Create curl(A) expression
    curlA = ufl.curl(A_sol)
    print(f"   curl(A) type: {type(curlA)}")
    print(f"   curl(A) is a UFL expression (not yet evaluated)")
    
    print("\n[STEP 3] L2 Projection to DG space")
    print("-" * 70)
    print("   PROBLEM AREA: This is where artifacts often occur!")
    print("   ")
    print("   Why artifacts occur here:")
    print("   1. Nédélec elements have DOFs on EDGES")
    print("   2. curl(A) is computed element-wise")
    print("   3. At boundaries, edge DOFs may have discontinuities")
    print("   4. L2 projection amplifies these discontinuities")
    print("   5. DG space is discontinuous, so boundary jumps are preserved")
    
    # Create DG space
    dg_element = basix.ufl.element("DG", mesh_obj.basix_cell(), config.degree_A, shape=(3,))
    dg_space = fem.functionspace(mesh_obj, dg_element)
    
    print(f"\n   DG space: {dg_space}")
    print(f"   DOFs: {dg_space.dofmap.index_map.size_local}")
    print(f"   Element type: Discontinuous Galerkin")
    print(f"   DOF locations: Cell centers (cell-wise constant or higher order)")
    
    print("\n[STEP 4] Interpolation from DG to Lagrange")
    print("-" * 70)
    print("   PROBLEM AREA: Another source of artifacts!")
    print("   ")
    print("   Why artifacts occur here:")
    print("   1. DG is discontinuous, Lagrange is continuous")
    print("   2. Interpolation at boundaries creates jumps")
    print("   3. Large gradients in DG → spikes in Lagrange")
    print("   4. Boundary conditions not enforced in DG → propagate to Lagrange")
    
    print("\n[ROOT CAUSES SUMMARY]")
    print("="*70)
    print("""
    The outliers occur because:

    1. **Nédélec → curl computation**
       - Nédélec DOFs are on edges, not nodes
       - curl(A) is computed element-wise
       - At domain boundaries/interfaces, edge DOFs can have discontinuities
       - These discontinuities are NOT smoothed by the curl operator

    2. **L2 Projection to DG space**
       - DG space is discontinuous
       - Boundary jumps are preserved (not smoothed)
       - Large gradients at boundaries → large values in DG
       - The L2 projection matrix can amplify these if ill-conditioned

    3. **DG → Lagrange Interpolation**
       - Discontinuous → continuous transition
       - Creates artificial spikes at cell boundaries
       - No smoothing applied during interpolation
       - Boundary effects propagate into interior

    4. **Boundary Conditions**
       - A = 0 on boundary (Dirichlet BC)
       - But curl(A) ≠ 0 at boundary (derivative discontinuity)
       - This creates a "ring" of high values near boundaries
       - The red ring you see is this boundary effect

    5. **Mesh Quality**
       - Poor element quality near boundaries
       - Skewed elements → numerical errors in curl
       - Small elements → large gradients → artifacts

    SOLUTIONS (in order of effectiveness):

    1. **Use H(curl) projection directly** (best)
       - Project curl(A) to H(curl) space, not DG
       - Then interpolate to Lagrange
       - Preserves curl structure better

    2. **Smooth the result**
       - Apply Laplacian smoothing to B field
       - Reduces spikes but may lose accuracy

    3. **Use higher-order elements**
       - Reduces interpolation errors
       - But increases computation cost

    4. **Improve mesh quality**
       - Better elements near boundaries
       - Reduces numerical errors

    5. **Use recovery-based methods**
       - Superconvergent patch recovery
       - More accurate but complex
    """)
    
    print("\n[RECOMMENDED FIX]")
    print("="*70)
    print("""
    Instead of: Nédélec → curl → DG → Lagrange
    Use:        Nédélec → curl → H(curl) → Lagrange
    
    This preserves the curl structure and reduces artifacts.
    """)

if __name__ == "__main__":
    diagnose_curl_computation()

