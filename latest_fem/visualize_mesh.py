import numpy as np
import matplotlib.pyplot as plt
from dolfinx.io import gmshio
from mpi4py import MPI

print("="*70)
print(" MOTOR MESH VISUALIZATION")
print("="*70)

mesh, ct, *rest = gmshio.read_from_msh("meshes/three_phase.msh", MPI.COMM_WORLD, rank=0, gdim=2)
print(f"\n✅ Mesh loaded: {mesh.topology.index_map(2).size_global} cells, {mesh.topology.index_map(0).size_global} vertices")

coords = mesh.geometry.x[:, :2]
cells = mesh.geometry.dofmap.reshape(-1, 3)

domains = {
    "Air": (1,), "AirGap": (2, 3), "Al": (4,), "Rotor": (5,), 
    "Stator": (6,), "Cu": (7, 8, 9, 10, 11, 12),
    "PM": (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
}

domain_colors = {
    "Air": "#F0F8FF", "AirGap": "#FFFACD", "Al": "#D3D3D3",
    "Rotor": "#CD853F", "Stator": "#708090", "Cu": "#FF6347", "PM": "#DC143C"
}

fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect('equal')

r_max = np.max(np.sqrt(coords[:, 0]**2 + coords[:, 1]**2))
limit = r_max * 0.75
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)

print("🎨 Creating visualization...")

for domain_name, markers in domains.items():
    if domain_name == "Air":
        continue
    color = domain_colors.get(domain_name, "#CCCCCC")
    for marker in markers:
        domain_cells = ct.find(marker)
        if len(domain_cells) > 0:
            for cell_idx in domain_cells:
                cell_coords = coords[cells[cell_idx]]
                r_cell = np.mean(np.sqrt(cell_coords[:, 0]**2 + cell_coords[:, 1]**2))
                if r_cell < limit:
                    triangle = plt.Polygon(cell_coords, facecolor=color, edgecolor='black',
                                          linewidth=0.1, alpha=0.9)
                    ax.add_patch(triangle)

ax.set_title("PMSM Motor Geometry", fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel("X [m]", fontsize=14)
ax.set_ylabel("Y [m]", fontsize=14)
ax.grid(False)

legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor=domain_colors["Rotor"], edgecolor='black', linewidth=1, label='Rotor'),
    plt.Rectangle((0,0),1,1, facecolor=domain_colors["PM"], edgecolor='black', linewidth=1, label='Permanent Magnets'),
    plt.Rectangle((0,0),1,1, facecolor=domain_colors["AirGap"], edgecolor='black', linewidth=1, label='Air Gap'),
    plt.Rectangle((0,0),1,1, facecolor=domain_colors["Stator"], edgecolor='black', linewidth=1, label='Stator'),
    plt.Rectangle((0,0),1,1, facecolor=domain_colors["Cu"], edgecolor='black', linewidth=1, label='Copper Coils'),
    plt.Rectangle((0,0),1,1, facecolor=domain_colors["Al"], edgecolor='black', linewidth=1, label='Aluminum')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)

ax.set_facecolor('#FFFFFF')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig("motor_mesh.png", dpi=200, bbox_inches='tight', facecolor='white')
print(f"\n💾 Saved: motor_mesh.png")

element_sizes = []
for cell_idx in range(len(cells)):
    cell_coords = coords[cells[cell_idx]]
    edges = [
        np.linalg.norm(cell_coords[1] - cell_coords[0]),
        np.linalg.norm(cell_coords[2] - cell_coords[1]),
        np.linalg.norm(cell_coords[0] - cell_coords[2])
    ]
    element_sizes.append(np.mean(edges))

element_sizes = np.array(element_sizes)

print(f"\n📊 Mesh statistics:")
print(f"   Min element size: {element_sizes.min()*1000:.4f} mm (finest)")
print(f"   Max element size: {element_sizes.max()*1000:.4f} mm (coarsest)")
print(f"   Mean element size: {element_sizes.mean()*1000:.4f} mm")

print("\n" + "="*70)
print(" ✅ Motor visualization complete!")
print("="*70)

