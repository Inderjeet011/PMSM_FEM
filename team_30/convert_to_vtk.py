#!/usr/bin/env python3
"""
Convert TEAM 30 BP results to VTK format for ParaView
"""

import numpy as np
import pyvista as pv
from dolfinx import io, fem
from mpi4py import MPI

print("="*70)
print(" Converting TEAM 30 Results to VTK Format")
print("="*70)

# Load mesh
print("\n📖 Loading mesh...")
with io.XDMFFile(MPI.COMM_WORLD, "meshes/three_phase3D.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="Cell_markers")

print(f"✅ Mesh: {mesh.topology.index_map(2).size_global} cells")

# Get mesh coordinates and connectivity
x = mesh.geometry.x
cells = mesh.geometry.dofmap.reshape(-1, 3)

# Create PyVista mesh
points = x[:, :2]  # 2D points
points_3d = np.column_stack([points, np.zeros(len(points))])  # Add z=0

# Create cells array for PyVista (prepend cell size to each cell)
cells_pv = np.hstack([np.full((len(cells), 1), 3), cells]).flatten()

# Create PyVista UnstructuredGrid
grid = pv.UnstructuredGrid(cells_pv, np.full(len(cells), pv.CellType.TRIANGLE), points_3d)

print(f"✅ Created PyVista grid: {grid.n_cells} cells, {grid.n_points} points")

# Add cell tags
grid.cell_data["region"] = ct.values

# Function to extract data from BP files
def extract_from_bp(bp_dir, var_name):
    """Extract variable data from BP directory"""
    import adios2
    
    data_all_timesteps = []
    times = []
    
    try:
        # Create ADIOS2 engine
        adios = adios2.ADIOS(MPI.COMM_WORLD)
        io_adios = adios.DeclareIO("ReadIO")
        engine = io_adios.Open(bp_dir, adios2.Mode.Read)
        
        step = 0
        while engine.BeginStep() == adios2.StepStatus.OK:
            var = io_adios.InquireVariable(var_name)
            if var:
                data = np.zeros(var.Shape()[0], dtype=np.float64)
                engine.Get(var, data, adios2.Mode.Sync)
                data_all_timesteps.append(data.copy())
                times.append(step * 0.00016666666666666666)  # dt from solver
            engine.EndStep()
            step += 1
        
        engine.Close()
        return data_all_timesteps, times
    except Exception as e:
        print(f"⚠️  Could not read {bp_dir}: {e}")
        return None, None

# Try to load Az data
print("\n📖 Reading Az data from BP files...")
Az_data, times = extract_from_bp("TEAM30_0.0_three/Az.bp", "Az")

if Az_data:
    print(f"✅ Found {len(Az_data)} timesteps")
    
    # Save each timestep
    for i, (data, t) in enumerate(zip(Az_data, times)):
        grid_copy = grid.copy()
        
        # Interpolate point data to cell centers if needed
        if len(data) == grid.n_points:
            grid_copy.point_data["Az"] = data
        elif len(data) == grid.n_cells:
            grid_copy.cell_data["Az"] = data
        else:
            # Pad or truncate as needed
            if len(data) < grid.n_points:
                padded = np.zeros(grid.n_points)
                padded[:len(data)] = data
                grid_copy.point_data["Az"] = padded
            else:
                grid_copy.point_data["Az"] = data[:grid.n_points]
        
        # Compute B = curl(Az)
        if "Az" in grid_copy.point_data:
            # Compute gradient
            grad = grid_copy.compute_derivative(scalars="Az", gradient="grad_Az")
            
            # B = curl(Az) = (dAz/dy, -dAz/dx, 0) in 2D
            if "grad_Az" in grad.point_data:
                grad_az = grad.point_data["grad_Az"]
                Bx = grad_az[:, 1]   # dAz/dy
                By = -grad_az[:, 0]  # -dAz/dx
                Bz = np.zeros_like(Bx)
                B = np.column_stack([Bx, By, Bz])
                grid_copy.point_data["B"] = B
                grid_copy.point_data["B_magnitude"] = np.linalg.norm(B, axis=1)
        
        # Save to VTK
        filename = f"output/Az_t{i:04d}.vtu"
        grid_copy.save(filename)
        
        if i == 0 or i == len(Az_data)-1:
            print(f"  Saved: {filename} (t={t:.6f} s)")
    
    print(f"\n✅ Saved {len(Az_data)} VTK files to output/Az_t*.vtu")
else:
    print("❌ Could not read BP files. Trying alternate method...")
    
    # Alternative: Just save mesh with regions
    grid.save("output/mesh_with_regions.vtu")
    print("✅ Saved mesh to output/mesh_with_regions.vtu")

print("\n" + "="*70)
print(" To visualize in ParaView:")
print("="*70)
print("1. Open ParaView")
print("2. File → Open → output/Az_t0000.vtu")  
print("3. Select 'B_magnitude' from the coloring dropdown")
print("4. Click Apply")
print("5. Use the play button to animate through timesteps")
print("="*70)

