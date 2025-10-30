#!/usr/bin/env python3
"""
Air Gap Magnetic Field Extractor
=================================
Extract and visualize B-field ONLY in the air gap region from simulation results

Features:
- Extracts B-field from mixed formulation results
- Masks to show only air gap region
- Computes magnitude and statistics
- Exports for ParaView visualization
"""

import h5py
import numpy as np
from dolfinx.io import gmshio, XDMFFile
from dolfinx import fem
from mpi4py import MPI
import ufl


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for field extraction"""
    MESH_FILE = "motor.msh"
    RESULTS_FILE = "results_2d_mixed.h5"
    OUTPUT_FILE = "airgap_B_only.xdmf"
    
    # Domain tags
    AIRGAP_INNER = 5
    AIRGAP_OUTER = 6
    
    # Time stepping
    TIMESTEP_MS = 2.0  # milliseconds


# ============================================================================
# AIR GAP FIELD EXTRACTOR CLASS
# ============================================================================

class AirGapFieldExtractor:
    """Extract and analyze magnetic field in air gap"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        
        # Storage
        self.mesh = None
        self.ct = None
        self.airgap_cells = None
        
        # Function spaces
        self.V_scalar = None  # Scalar CG space for Az
        self.V_vector = None  # Vector DG space for B
        self.V_mag = None     # Scalar DG space for |B|
        
        # Functions
        self.Az = None
        self.B_full = None
        self.B_airgap = None
        self.B_mag_func = None
        
    def load_mesh(self):
        """Load mesh and identify air gap cells"""
        print("=" * 70)
        print(" EXTRACTING B-FIELD IN AIR GAP")
        print("=" * 70)
        print(f"\n📖 Loading mesh: {self.config.MESH_FILE}")
        
        self.mesh, self.ct, *rest = gmshio.read_from_msh(
            self.config.MESH_FILE, MPI.COMM_WORLD, rank=0, gdim=2
        )
        
        print(f"   ✅ {self.ct.values.size} cells loaded")
        
        # Find air gap cells
        airgap_cells_inner = self.ct.find(self.config.AIRGAP_INNER)
        airgap_cells_outer = self.ct.find(self.config.AIRGAP_OUTER)
        self.airgap_cells = np.concatenate([airgap_cells_inner, airgap_cells_outer])
        
        print(f"\n🔍 Air gap cells identified:")
        print(f"   Inner layer: {len(airgap_cells_inner)} cells")
        print(f"   Outer layer: {len(airgap_cells_outer)} cells")
        print(f"   Total:       {len(self.airgap_cells)} cells")
        
    def setup_function_spaces(self):
        """Setup function spaces for field computation"""
        print("\n🎯 Setting up function spaces...")
        
        # Scalar space for Az (matches solver output)
        self.V_scalar = fem.functionspace(self.mesh, ("CG", 1))
        
        # Vector space for B-field (DG for discontinuous field)
        self.V_vector = fem.functionspace(self.mesh, ("DG", 0, (2,)))
        
        # Scalar space for |B| magnitude
        self.V_mag = fem.functionspace(self.mesh, ("DG", 0))
        
        # Create functions
        self.Az = fem.Function(self.V_scalar)
        self.B_full = fem.Function(self.V_vector)
        self.B_airgap = fem.Function(self.V_vector)
        self.B_mag_func = fem.Function(self.V_mag)
        
        # Set names for ParaView
        self.B_airgap.name = "B_airgap"
        self.B_mag_func.name = "B_magnitude"
        
        print("   ✅ Function spaces created")
        
    def compute_b_field(self):
        """Compute B = curl(Az) from magnetic vector potential"""
        # B = curl(Az) = (∂Az/∂y, -∂Az/∂x)
        B_expr = fem.Expression(
            ufl.as_vector((self.Az.dx(1), -self.Az.dx(0))),
            self.V_vector.element.interpolation_points
        )
        self.B_full.interpolate(B_expr)
        
    def mask_to_airgap(self):
        """Mask B-field to show only air gap region"""
        # Zero out all cells
        self.B_airgap.x.array[:] = 0.0
        
        # Copy B-field only in air gap cells
        for cell in self.airgap_cells:
            idx = 2 * cell  # Vector function has 2 components per cell
            self.B_airgap.x.array[idx:idx+2] = self.B_full.x.array[idx:idx+2]
        
    def compute_magnitude(self):
        """Compute |B| = sqrt(Bx^2 + By^2)"""
        # Extract components
        Bx = self.B_airgap.sub(0).collapse().x.array
        By = self.B_airgap.sub(1).collapse().x.array
        
        # Compute magnitude
        B_mag = np.sqrt(Bx**2 + By**2)
        self.B_mag_func.x.array[:] = B_mag
        
        return B_mag
    
    def extract_all_timesteps(self):
        """Extract B-field for all timesteps"""
        print(f"\n⏱️  Processing timesteps from: {self.config.RESULTS_FILE}")
        
        with h5py.File(self.config.RESULTS_FILE, "r") as h5f:
            # Access the function group (Az is first component)
            f0_group = h5f["Function"]["f_0"]
            timesteps = sorted([k for k in f0_group.keys() if k.startswith("0_")])
            n_steps = len(timesteps)
            
            print(f"   ✅ Found {n_steps} timesteps")
            
            # Open output file
            with XDMFFile(self.mesh.comm, self.config.OUTPUT_FILE, "w") as xdmf:
                xdmf.write_mesh(self.mesh)
                
                B_max_list = []
                B_mean_list = []
                
                print("\n📊 Processing:")
                
                for i, ts in enumerate(timesteps):
                    # Load Az for this timestep
                    data = f0_group[ts][:]
                    self.Az.x.array[:] = data.flatten()[:len(self.Az.x.array)]
                    
                    # Compute B-field
                    self.compute_b_field()
                    
                    # Mask to air gap
                    self.mask_to_airgap()
                    
                    # Compute magnitude
                    B_mag = self.compute_magnitude()
                    
                    # Time in milliseconds
                    time_ms = (i + 1) * self.config.TIMESTEP_MS
                    
                    # Write to output
                    xdmf.write_function(self.B_airgap, time_ms)
                    xdmf.write_function(self.B_mag_func, time_ms)
                    
                    # Statistics (only in air gap)
                    B_airgap_vals = B_mag[self.airgap_cells]
                    B_max_list.append(B_airgap_vals.max())
                    B_mean_list.append(B_airgap_vals.mean())
                    
                    print(f"   Step {i+1:2d}/{n_steps}  "
                          f"t={time_ms:5.1f} ms  "
                          f"max|B|={B_max_list[-1]:.4f} T  "
                          f"mean|B|={B_mean_list[-1]:.4f} T")
                
                # Final statistics
                print("\n📈 Air Gap B-Field Statistics:")
                print(f"   Cells analyzed:     {len(self.airgap_cells)}")
                print(f"   max|B| (all time):  {max(B_max_list):.4f} T")
                print(f"   mean|B| (all time): {np.mean(B_mean_list):.4f} T")
                print(f"   min|B| (all time):  {min(B_max_list):.4f} T")
        
        print(f"\n✅ Extracted to: {self.config.OUTPUT_FILE}")
        
    def run(self):
        """Execute full extraction workflow"""
        self.load_mesh()
        self.setup_function_spaces()
        self.extract_all_timesteps()
        
        print("\n" + "=" * 70)
        print(" ✅ EXTRACTION COMPLETE")
        print("=" * 70)
        print("\n💡 Next steps:")
        print(f"   1. Open ParaView: paraview {self.config.OUTPUT_FILE}")
        print("   2. Select 'B_magnitude' in the dropdown")
        print("   3. Click PLAY ▶️  to see field rotation animation")
        print("   4. Use 'Glyph' filter on 'B_airgap' to show field vectors")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    extractor = AirGapFieldExtractor()
    extractor.run()

