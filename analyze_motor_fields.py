#!/usr/bin/env python3
"""
Extract and analyze B-field around different motor components
Direct analysis - no ParaView needed!

Usage: python analyze_motor_fields.py
"""

import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def load_mesh_and_fields():
    """Load mesh, cell tags, and field data"""
    h5_path = Path('results/3d/av_solver.h5')
    
    print("Loading data...")
    with h5py.File(h5_path, 'r') as f:
        # Mesh
        vertices = f['Mesh/mesh/geometry'][:]
        cells = f['Mesh/mesh/topology'][:]
        
        # Cell tags (which cell belongs to which region)
        cell_tags = None
        if 'MeshTags/mesh_tags/Values' in f:
            cell_tags = f['MeshTags/mesh_tags/Values'][:]
        
        # Fields (first timestep)
        fields = {}
        if 'Function' in f:
            for field_name in f['Function'].keys():
                timesteps = list(f[f'Function/{field_name}'].keys())
                if timesteps:
                    data = f[f'Function/{field_name}/{timesteps[0]}'][:]
                    fields[field_name] = data
        
        # MaterialID
        if 'Function/MaterialID' in f:
            timesteps = list(f['Function/MaterialID'].keys())
            if timesteps:
                fields['MaterialID'] = f[f'Function/MaterialID/{timesteps[0]}'][:]
    
    print(f"  Loaded {len(vertices):,} vertices, {len(cells):,} cells")
    return vertices, cells, cell_tags, fields

def get_cell_centers(vertices, cells):
    """Compute cell centers"""
    cell_centers = []
    for cell in cells:
        cell_vertices = vertices[cell]
        center = np.mean(cell_vertices, axis=0)
        cell_centers.append(center)
    return np.array(cell_centers)

def analyze_by_region(vertices, cells, cell_tags, fields):
    """Analyze B-field for each motor component"""
    
    print("\n" + "="*70)
    print("ANALYZING B-FIELD BY MOTOR COMPONENT")
    print("="*70)
    
    # Material ID mapping
    material_map = {
        1: "Air",
        2: "AirGap", 
        3: "Rotor",
        4: "Stator",
        5: "Coils",
        6: "Magnets",
        7: "Aluminium"
    }
    
    # Get cell centers
    cell_centers = get_cell_centers(vertices, cells)
    
    # Get B-field (use cell data if available, otherwise interpolate from points)
    B_magnitude = None
    B_vector = None
    
    if 'B_dg' in fields:
        # Cell-centered B-field (DG0)
        B_dg_data = fields['B_dg']
        if len(B_dg_data.shape) == 2 and B_dg_data.shape[1] == 3:
            B_dg = B_dg_data
        else:
            B_dg = B_dg_data.reshape(-1, 3)
        B_magnitude = np.linalg.norm(B_dg, axis=1)
        B_vector = B_dg
        print(f"Using cell-centered B-field (B_dg): shape={B_dg.shape}, B_mag shape={B_magnitude.shape}")
    elif 'B_Magnitude' in fields and len(fields['B_Magnitude']) == len(cells):
        B_magnitude = fields['B_Magnitude']
        if 'B' in fields:
            B_vector = fields['B'].reshape(-1, 3)
        print("Using cell B-field data")
    elif 'B_Magnitude' in fields:
        # Point data - need to map to cells (average of cell vertices)
        print("Mapping point B-field to cells...")
        B_mag_points = fields['B_Magnitude']
        B_magnitude = []
        if 'B' in fields:
            B_points = fields['B'].reshape(-1, 3)
            B_vector_list = []
        
        for cell in cells:
            cell_B_mag = np.mean(B_mag_points[cell])
            B_magnitude.append(cell_B_mag)
            if 'B' in fields:
                cell_B_vec = np.mean(B_points[cell], axis=0)
                B_vector_list.append(cell_B_vec)
        
        B_magnitude = np.array(B_magnitude)
        if 'B' in fields:
            B_vector = np.array(B_vector_list)
        print("Mapped point data to cells")
    
    if B_magnitude is None:
        print("ERROR: Could not find B-field data!")
        return None
    
    # Analyze by MaterialID
    if 'MaterialID' not in fields or len(fields['MaterialID']) != len(cells):
        print("WARNING: MaterialID not available or size mismatch")
        print("Using cell tags instead...")
        if cell_tags is not None:
            # Map cell tags to materials
            tag_to_material = {
                1: "Air", 2: "AirGap", 3: "AirGap",
                4: "Aluminium", 5: "Rotor", 6: "Stator"
            }
            # Coils: 7-12, Magnets: 13-22
            for tag in range(7, 13):
                tag_to_material[tag] = "Coils"
            for tag in range(13, 23):
                tag_to_material[tag] = "Magnets"
            
            material_ids = np.array([tag_to_material.get(int(tag), "Unknown") for tag in cell_tags])
        else:
            print("ERROR: No MaterialID or cell tags available!")
            return None
    else:
        material_ids = fields['MaterialID']
        # Convert numeric IDs to names
        material_names = np.array([material_map.get(int(mid), f"Unknown({int(mid)})") 
                                   for mid in material_ids])
    
    # Group by material
    results = {}
    for material_name in material_map.values():
        if material_name == "Air":  # Skip outer air
            continue
            
        # Find cells belonging to this material
        if 'MaterialID' in fields and len(fields['MaterialID']) == len(cells):
            # Use numeric comparison
            material_id_num = [k for k, v in material_map.items() if v == material_name]
            if material_id_num:
                mask = (material_ids == material_id_num[0])
            else:
                mask = np.zeros(len(cells), dtype=bool)
        else:
            if 'material_names' in locals():
                mask = material_names == material_name
            else:
                mask = np.zeros(len(cells), dtype=bool)
        
        if np.sum(mask) == 0:
            continue
        
        # Extract data for this material
        # Ensure mask is 1D boolean array
        if mask.ndim > 1:
            mask = mask.flatten()
        mask = mask.astype(bool)
        
        # Ensure B_magnitude is 1D
        if B_magnitude.ndim > 1:
            B_magnitude = B_magnitude.flatten()
        
        material_B_mag = B_magnitude[mask]
        material_centers = cell_centers[mask]
        if B_vector is not None:
            if B_vector.ndim == 2:
                material_B_vec = B_vector[mask]
            else:
                material_B_vec = None
        else:
            material_B_vec = None
        
        # Compute statistics
        B_nonzero = material_B_mag[material_B_mag > 1e-6]
        
        results[material_name] = {
            'n_cells': int(np.sum(mask)),
            'B_magnitude': material_B_mag,
            'B_nonzero': B_nonzero,
            'centers': material_centers,
            'B_vector': material_B_vec,
            'mean': float(np.mean(B_nonzero)) if len(B_nonzero) > 0 else 0.0,
            'max': float(np.max(material_B_mag)),
            'min': float(np.min(material_B_mag)),
            'median': float(np.median(B_nonzero)) if len(B_nonzero) > 0 else 0.0,
            'std': float(np.std(B_nonzero)) if len(B_nonzero) > 0 else 0.0,
            'nonzero_fraction': len(B_nonzero) / len(material_B_mag) if len(material_B_mag) > 0 else 0.0
        }
        
        # Print summary
        print(f"\n{material_name.upper()}:")
        print(f"  Cells: {results[material_name]['n_cells']:,}")
        if len(B_nonzero) > 0:
            print(f"  B-field: max={results[material_name]['max']:.4f} T, "
                  f"mean={results[material_name]['mean']:.4f} T, "
                  f"median={results[material_name]['median']:.4f} T")
            print(f"  Non-zero field: {results[material_name]['nonzero_fraction']*100:.1f}% of cells")
        else:
            print(f"  B-field: all values near zero")
    
    return results

def create_visualizations(results):
    """Create visualizations for each component"""
    
    if results is None:
        return
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Summary statistics plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Extract data
    materials = list(results.keys())
    means = [results[m]['mean'] for m in materials]
    maxs = [results[m]['max'] for m in materials]
    medians = [results[m]['median'] for m in materials]
    n_cells = [results[m]['n_cells'] for m in materials]
    
    # Plot 1: Mean B-field by component
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(materials)), means, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(materials)))
    ax1.set_xticklabels(materials, rotation=45, ha='right')
    ax1.set_ylabel('Mean B-field (T)')
    ax1.set_title('Mean B-field by Motor Component')
    ax1.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, means)):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Max B-field by component
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(materials)), maxs, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(materials)))
    ax2.set_xticklabels(materials, rotation=45, ha='right')
    ax2.set_ylabel('Max B-field (T)')
    ax2.set_title('Maximum B-field by Motor Component')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars2, maxs)):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: B-field distribution (histogram)
    ax3 = axes[1, 0]
    for material in materials:
        B_nonzero = results[material]['B_nonzero']
        if len(B_nonzero) > 0:
            ax3.hist(B_nonzero, bins=30, alpha=0.5, label=material, edgecolor='black')
    ax3.set_xlabel('B-field Magnitude (T)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('B-field Distribution by Component')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_yscale('log')
    
    # Plot 4: Number of cells
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(materials)), n_cells, color='lightgreen', alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(materials)))
    ax4.set_xticklabels(materials, rotation=45, ha='right')
    ax4.set_ylabel('Number of Cells')
    ax4.set_title('Mesh Resolution by Component')
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars4, n_cells)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:,}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    plt.tight_layout()
    plt.savefig('motor_component_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: motor_component_analysis.png")
    
    # 2. Detailed field maps for each component (XY slices)
    n_components = len(materials)
    n_cols = 3
    n_rows = (n_components + n_cols - 1) // n_cols
    
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_components == 1:
        axes2 = [axes2]
    else:
        axes2 = axes2.flatten()
    
    for idx, material in enumerate(materials):
        ax = axes2[idx]
        centers = results[material]['centers']
        B_mag = results[material]['B_magnitude']
        
        # Take XY slice (middle Z)
        z_mid = np.mean(centers[:, 2])
        z_tol = 0.02
        mask = np.abs(centers[:, 2] - z_mid) < z_tol
        
        if np.sum(mask) > 0:
            scatter = ax.scatter(centers[mask, 0], centers[mask, 1], 
                               c=B_mag[mask], cmap='coolwarm', s=30, 
                               edgecolors='black', linewidth=0.3, vmin=0, vmax=np.max(B_mag))
            plt.colorbar(scatter, ax=ax, label='B (T)')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{material}\nMax={results[material]["max"]:.3f}T, Mean={results[material]["mean"]:.3f}T')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{material}\nNo data in slice', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(material)
    
    # Hide unused subplots
    for idx in range(n_components, len(axes2)):
        axes2[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('motor_component_fields.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: motor_component_fields.png")
    
    # 3. Export data to CSV
    print("\nExporting data to CSV...")
    import csv
    
    with open('motor_field_statistics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Component', 'N_Cells', 'B_Max_T', 'B_Mean_T', 'B_Median_T', 
                         'B_Std_T', 'B_Min_T', 'NonZero_Fraction'])
        for material in materials:
            r = results[material]
            writer.writerow([
                material,
                r['n_cells'],
                f"{r['max']:.6f}",
                f"{r['mean']:.6f}",
                f"{r['median']:.6f}",
                f"{r['std']:.6f}",
                f"{r['min']:.6f}",
                f"{r['nonzero_fraction']:.4f}"
            ])
    print("✓ Saved: motor_field_statistics.csv")
    
    # 4. Print detailed summary
    print("\n" + "="*70)
    print("DETAILED SUMMARY")
    print("="*70)
    for material in materials:
        r = results[material]
        print(f"\n{material.upper()}:")
        print(f"  Cells: {r['n_cells']:,}")
        print(f"  B-field Statistics:")
        print(f"    Max:    {r['max']:.6f} T")
        print(f"    Mean:   {r['mean']:.6f} T")
        print(f"    Median: {r['median']:.6f} T")
        print(f"    Std:    {r['std']:.6f} T")
        print(f"    Min:    {r['min']:.6f} T")
        print(f"    Non-zero: {r['nonzero_fraction']*100:.1f}% of cells")

def main():
    print("="*70)
    print("MOTOR COMPONENT FIELD ANALYSIS")
    print("="*70)
    
    vertices, cells, cell_tags, fields = load_mesh_and_fields()
    results = analyze_by_region(vertices, cells, cell_tags, fields)
    create_visualizations(results)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nOutput files:")
    print("  - motor_component_analysis.png: Statistics plots")
    print("  - motor_component_fields.png: Field maps for each component")
    print("  - motor_field_statistics.csv: Data table")
    print("="*70)

if __name__ == '__main__':
    main()

