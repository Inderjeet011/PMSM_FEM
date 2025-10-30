#!/usr/bin/env python3
"""
Motor Mesh Viewer
=================
Visualize and analyze PM motor mesh created by pm_mesh_generator_2d.py

Features:
- Domain color visualization
- Mesh structure display
- Node density plots
"""

import gmsh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys


# ============================================================================
# CONFIGURATION
# ============================================================================

# Reference radii for motor geometry (in meters)
RADII = {
    'Rotor': 0.030,
    'PM': 0.038,
    'Gap': 0.040,
    'Coil': 0.050,
    'Stator': 0.057,
    'Outer': 0.090
}

# Domain colors for visualization
DOMAIN_COLORS = {
    1: ('OUTER_AIR', '#e0f7fa'),      # Light cyan
    2: ('ROTOR', '#b0bec5'),          # Gray
    3: ('PM_N', '#ffcdd2'),           # Light red (North)
    4: ('PM_S', '#bbdefb'),           # Light blue (South)
    5: ('AIR_GAP', '#fff9c4'),        # Light yellow
    6: ('AIR_GAP_OUTER', '#fff59d'),  # Yellow
    7: ('STATOR', '#c5e1a5'),         # Light green
    8: ('COIL_0', '#ffe0b2'),         # Light orange
    9: ('COIL_1', '#ffccbc'),
    10: ('COIL_2', '#f8bbd0'),
    11: ('COIL_3', '#e1bee7'),
    12: ('COIL_4', '#d1c4e9'),
    13: ('COIL_5', '#c5cae9'),
}


# ============================================================================
# MESH VIEWER CLASS
# ============================================================================

class MotorMeshViewer:
    """Interactive motor mesh viewer and analyzer"""
    
    def __init__(self, mesh_file="../motor.msh"):
        self.mesh_file = mesh_file
        self.node_coords = None
        self.element_types = None
        self.element_tags = None
        self.element_node_tags = None
        self.physical_groups = None
        self.qualities = None
        
    def load_mesh(self):
        """Load mesh file"""
        print("=" * 70)
        print(" MOTOR MESH VIEWER")
        print("=" * 70)
        print(f"\n📂 Loading mesh file: {self.mesh_file}")
        
        gmsh.initialize()
        try:
            gmsh.open(self.mesh_file)
        except:
            print(f"❌ Error: Could not find {self.mesh_file}")
            print("   Please run pm_motor_mesh_generator.py first!")
            gmsh.finalize()
            sys.exit(1)
        
        print("   ✅ Mesh loaded successfully")
        
    def get_statistics(self):
        """Extract mesh statistics"""
        print("\n📊 Mesh Statistics:")
        print("-" * 70)
        
        # Nodes
        node_tags, self.node_coords, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)
        print(f"   Total nodes:     {n_nodes:,}")
        
        # Elements
        self.element_types, self.element_tags, self.element_node_tags = \
            gmsh.model.mesh.getElements(dim=2)
        n_elements = sum(len(tags) for tags in self.element_tags)
        print(f"   Total elements:  {n_elements:,}")
        
        # Element types
        for elem_type, tags in zip(self.element_types, self.element_tags):
            elem_name = gmsh.model.mesh.getElementProperties(elem_type)[0]
            print(f"      {elem_name}: {len(tags):,}")
        
    def get_physical_groups(self):
        """Get physical group information"""
        print("\n🏷️  Physical Groups (Domains):")
        print("-" * 70)
        
        self.physical_groups = gmsh.model.getPhysicalGroups(dim=2)
        
        for dim, tag in self.physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            
            # Count elements
            n_elem = 0
            for entity in entities:
                elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim, entity)
                n_elem += sum(len(t) for t in elem_tags)
            
            print(f"   [{tag:3d}] {name:20s}: {len(entities):3d} surfaces, {n_elem:6,d} elements")
        
        # Boundary groups
        print("\n   Boundary markers:")
        boundary_groups = gmsh.model.getPhysicalGroups(dim=1)
        for dim, tag in boundary_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            print(f"   [{tag:3d}] {name:20s}: {len(entities):3d} curves")
        
    def analyze_quality(self):
        """Analyze mesh quality"""
        print("\n📐 Mesh Quality:")
        print("-" * 70)
        
        self.qualities = []
        for elem_type, elem_tags in zip(self.element_types, self.element_tags):
            elem_name = gmsh.model.mesh.getElementProperties(elem_type)[0]
            if "Triangle" in elem_name:
                elem_qualities = gmsh.model.mesh.getElementQualities(
                    elem_tags.tolist(), "minSICN"
                )
                self.qualities.extend(elem_qualities)
        
        if self.qualities:
            self.qualities = np.array(self.qualities)
            print(f"   Minimum quality:  {np.min(self.qualities):.4f}")
            print(f"   Maximum quality:  {np.max(self.qualities):.4f}")
            print(f"   Average quality:  {np.mean(self.qualities):.4f}")
            print(f"   Median quality:   {np.median(self.qualities):.4f}")
            print(f"   Elements < 0.3:   {np.sum(self.qualities < 0.3)} "
                  f"({100*np.sum(self.qualities < 0.3)/len(self.qualities):.1f}%)")
            print(f"   Elements < 0.5:   {np.sum(self.qualities < 0.5)} "
                  f"({100*np.sum(self.qualities < 0.5)/len(self.qualities):.1f}%)")
        
    def plot_domain_colors(self, ax):
        """Plot mesh with domain colors"""
        ax.set_aspect('equal')
        ax.set_title('Motor Mesh - Domain Colors', fontsize=14, fontweight='bold')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        
        coords = self.node_coords.reshape(-1, 3)
        
        # Plot each domain with its color
        for dim, tag in self.physical_groups:
            if tag in DOMAIN_COLORS:
                name, color = DOMAIN_COLORS[tag]
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                
                for entity in entities:
                    elem_types, elem_tags, elem_nodes = \
                        gmsh.model.mesh.getElements(dim, entity)
                    
                    for etype, etags, enodes in zip(elem_types, elem_tags, elem_nodes):
                        elem_name = gmsh.model.mesh.getElementProperties(etype)[0]
                        if "Triangle" in elem_name:
                            enodes = enodes.reshape(-1, 3) - 1  # 0-based indexing
                            for triangle_nodes in enodes:
                                triangle_coords = coords[triangle_nodes]
                                triangle = plt.Polygon(
                                    triangle_coords[:, :2],
                                    facecolor=color,
                                    edgecolor='black',
                                    linewidth=0.1,
                                    alpha=0.8
                                )
                                ax.add_patch(triangle)
        
        # Add reference circles
        for label, r in RADII.items():
            circle = Circle((0, 0), r, fill=False, edgecolor='red',
                          linestyle='--', linewidth=0.8, alpha=0.5)
            ax.add_patch(circle)
        
        ax.set_xlim(-0.095, 0.095)
        ax.set_ylim(-0.095, 0.095)
        ax.grid(True, alpha=0.3)
        
    def plot_mesh_structure(self, ax):
        """Plot mesh structure with all edges"""
        ax.set_aspect('equal')
        ax.set_title('Motor Mesh - Structure', fontsize=14, fontweight='bold')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        
        coords = self.node_coords.reshape(-1, 3)
        
        # Plot all triangles
        for elem_type, elem_tags, elem_nodes in zip(
            self.element_types, self.element_tags, self.element_node_tags
        ):
            elem_name = gmsh.model.mesh.getElementProperties(elem_type)[0]
            if "Triangle" in elem_name:
                elem_nodes = elem_nodes.reshape(-1, 3) - 1
                for triangle_nodes in elem_nodes:
                    triangle_coords = coords[triangle_nodes]
                    triangle = plt.Polygon(
                        triangle_coords[:, :2],
                        facecolor='lightgray',
                        edgecolor='black',
                        linewidth=0.2,
                        alpha=0.5
                    )
                    ax.add_patch(triangle)
        
        ax.set_xlim(-0.095, 0.095)
        ax.set_ylim(-0.095, 0.095)
        ax.grid(True, alpha=0.3)
        
    def plot_node_density(self, ax):
        """Plot node density"""
        ax.set_aspect('equal')
        ax.set_title('Motor Mesh - Node Density', fontsize=14, fontweight='bold')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        
        coords = self.node_coords.reshape(-1, 3)
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Plot nodes
        ax.scatter(x, y, c='blue', s=0.5, alpha=0.6)
        
        # Add reference circles with labels
        for label, r in RADII.items():
            circle = Circle((0, 0), r, fill=False, edgecolor='red',
                          linestyle='--', linewidth=1.0, alpha=0.7)
            ax.add_patch(circle)
            
            # Add label
            ax.text(r*0.707, r*0.707, label, fontsize=8, color='red',
                   ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_xlim(-0.095, 0.095)
        ax.set_ylim(-0.095, 0.095)
        ax.grid(True, alpha=0.3)
        
    def plot_quality_distribution(self):
        """Plot mesh quality distribution"""
        if len(self.qualities) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(self.qualities, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(self.qualities), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(self.qualities):.3f}')
        ax.axvline(np.median(self.qualities), color='green', linestyle='--',
                  linewidth=2, label=f'Median: {np.median(self.qualities):.3f}')
        
        ax.set_xlabel('Element Quality (SICN)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Mesh Quality Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        import os
        os.makedirs('../results', exist_ok=True)
        plt.savefig('../results/motor_mesh_quality.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved: ../results/motor_mesh_quality.png")
        
    def visualize(self):
        """Create all visualizations"""
        print("\n🎨 Creating visualizations...")
        
        # Main figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        self.plot_domain_colors(axes[0])
        self.plot_mesh_structure(axes[1])
        self.plot_node_density(axes[2])
        
        plt.tight_layout()
        plt.savefig('../results/motor_mesh_visualization.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved: ../results/motor_mesh_visualization.png")
        
    def run(self):
        """Execute full analysis"""
        self.load_mesh()
        self.get_statistics()
        self.get_physical_groups()
        self.visualize()
        
        gmsh.finalize()
        
        print("\n" + "=" * 70)
        print(" ✅ VISUALIZATION COMPLETE")
        print("=" * 70)
        print("\n💡 Options:")
        print("   1. View saved image: ../results/motor_mesh_visualization.png")
        print("   2. Launch interactive Gmsh GUI: add 'gmsh.fltk.run()' before finalize")
        print("   3. Mesh is ready for FEniCS simulation!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    viewer = MotorMeshViewer(mesh_file="../motor.msh")
    viewer.run()

