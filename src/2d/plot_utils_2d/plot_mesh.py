#!/usr/bin/env python3
"""
Mesh plotting utility for FEniCS/DOLFINx meshes created via Gmsh.

Usage examples:
  python3 plot_mesh.py --mesh /root/FEniCS/motor.msh --out /root/FEniCS/results/mesh_preview.png
  python3 plot_mesh.py --mesh /root/FEniCS/motor.msh --show --no-fill

Features:
- Colors cells by cell (physical) tags if available
- Optionally overlays facet (boundary) tags
- Saves to PNG or displays interactively
"""

import argparse
import os
from typing import Dict, Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

from dolfinx.io import gmshio
from mpi4py import MPI


def load_mesh_with_tags(mesh_path: str):
    mesh, ct, *rest = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, rank=0, gdim=2)
    ft = rest[0] if rest else None
    return mesh, ct, ft


def build_cell_polygons(mesh) -> np.ndarray:
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    coords = mesh.geometry.x

    polys = []
    for c in range(mesh.topology.index_map(tdim).size_local):
        vs = c2v.links(c)
        pts = coords[vs, :2]
        polys.append(pts)
    return polys


def build_facet_segments(mesh) -> np.ndarray:
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    f2v = mesh.topology.connectivity(tdim - 1, 0)
    coords = mesh.geometry.x
    segs = []
    for f in range(mesh.topology.index_map(tdim - 1).size_local):
        vs = f2v.links(f)
        if len(vs) >= 2:
            p = coords[vs, :2]
            # Break into segments (v0->v1, v1->v2, ...)
            for i in range(len(vs) - 1):
                segs.append([p[i], p[i + 1]])
            # Close if last not equal first and facet is polygonal
            if len(vs) > 2 and not np.allclose(p[0], p[-1]):
                segs.append([p[-1], p[0]])
    return segs


def get_cell_colors_by_tag(mesh, ct) -> Optional[np.ndarray]:
    if ct is None:
        return None
    tdim = mesh.topology.dim
    n_cells = mesh.topology.index_map(tdim).size_local
    colors = np.zeros(n_cells, dtype=float)
    # Map unique tag values to [0, 1]
    cell_to_tag: Dict[int, int] = {int(i): int(v) for i, v in zip(ct.indices, ct.values)}
    if not cell_to_tag:
        return None
    tags = np.array(list(set(cell_to_tag.values())), dtype=int)
    tags_sorted = np.sort(tags)
    tag_to_norm = {tag: (idx + 1) / (len(tags_sorted) + 1) for idx, tag in enumerate(tags_sorted)}
    for c in range(n_cells):
        tag = cell_to_tag.get(c)
        colors[c] = tag_to_norm.get(tag, 0.0)
    return colors


def get_realistic_facecolors(mesh, ct) -> Optional[List[Tuple[float, float, float, float]]]:
    if ct is None:
        return None
    tdim = mesh.topology.dim
    n_cells = mesh.topology.index_map(tdim).size_local
    # DomainTags mapping mirrored from solver_2d.py
    TAG_COLORS: Dict[int, Tuple[float, float, float, float]] = {
        1: (0.94, 0.96, 0.97, 1.0),  # OUTER_AIR - very light blue-gray
        5: (0.95, 0.97, 0.99, 1.0),  # AIRGAP_INNER
        6: (0.95, 0.97, 0.99, 1.0),  # AIRGAP_OUTER
        2: (0.70, 0.72, 0.75, 1.0),  # ROTOR - steel gray
        7: (0.55, 0.57, 0.60, 1.0),  # STATOR - darker iron
        3: (0.85, 0.20, 0.20, 1.0),  # PM_N - red
        4: (0.20, 0.35, 0.85, 1.0),  # PM_S - blue
        8: (0.98, 0.65, 0.15, 1.0),  # COIL A+
        11: (0.98, 0.85, 0.55, 1.0), # COIL A-
        9: (0.20, 0.75, 0.30, 1.0),  # COIL B+
        12: (0.60, 0.85, 0.65, 1.0), # COIL B-
        10: (0.75, 0.20, 0.75, 1.0), # COIL C+
        13: (0.88, 0.65, 0.88, 1.0), # COIL C-
    }
    cell_to_tag: Dict[int, int] = {int(i): int(v) for i, v in zip(ct.indices, ct.values)}
    if not cell_to_tag:
        return None
    default_color = (0.90, 0.92, 0.95, 1.0)
    facecolors: List[Tuple[float, float, float, float]] = []
    for c in range(n_cells):
        tag = cell_to_tag.get(c)
        facecolors.append(TAG_COLORS.get(tag, default_color))
    return facecolors


def plot_mesh(mesh_path: str, out_path: Optional[str], show: bool, no_fill: bool, draw_facets: bool, realistic: bool):
    mesh, ct, ft = load_mesh_with_tags(mesh_path)
    polys = build_cell_polygons(mesh)
    cell_colors = get_cell_colors_by_tag(mesh, ct)
    realistic_colors = get_realistic_facecolors(mesh, ct) if not no_fill and realistic else None

    fig, ax = plt.subplots(figsize=(8, 8))

    # Polygons (cells)
    poly_coll = PolyCollection(polys, edgecolors=(0, 0, 0, 0.35), linewidths=0.2,
                               facecolors='none' if no_fill else None, cmap='viridis')
    if not no_fill:
        if realistic_colors is not None:
            poly_coll.set_facecolor(realistic_colors)
        elif cell_colors is not None:
            poly_coll.set_array(cell_colors)
        else:
            poly_coll.set_facecolor((0.9, 0.9, 0.95, 1.0))
    ax.add_collection(poly_coll)

    # Facet overlay (optional)
    if draw_facets:
        segs = build_facet_segments(mesh)
        if len(segs):
            lc = LineCollection(segs, colors=(1.0, 0.2, 0.2, 0.6), linewidths=0.4)
            ax.add_collection(lc)

    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(os.path.basename(mesh_path))
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Colorbar only for normalized (non-realistic) coloring
    if not no_fill and realistic_colors is None and cell_colors is not None:
        cbar = fig.colorbar(poly_coll, ax=ax, shrink=0.8)
        cbar.set_label('Cell tag (normalized)')

    # Legend for realistic colors
    if not no_fill and realistic_colors is not None and ct is not None:
        import matplotlib.patches as mpatches
        used_tags = sorted(set(int(v) for v in ct.values))
        tag_labels = {
            1: 'Air (outer)', 5: 'Air-gap inner', 6: 'Air-gap outer',
            2: 'Rotor (steel)', 7: 'Stator (iron)',
            3: 'PM North', 4: 'PM South',
            8: 'Coil A+', 11: 'Coil A-', 9: 'Coil B+', 12: 'Coil B-', 10: 'Coil C+', 13: 'Coil C-'
        }
        TAG_COLORS = {  # same as above
            1: (0.94, 0.96, 0.97, 1.0), 5: (0.95, 0.97, 0.99, 1.0), 6: (0.95, 0.97, 0.99, 1.0),
            2: (0.70, 0.72, 0.75, 1.0), 7: (0.55, 0.57, 0.60, 1.0), 3: (0.85, 0.20, 0.20, 1.0),
            4: (0.20, 0.35, 0.85, 1.0), 8: (0.98, 0.65, 0.15, 1.0), 11: (0.98, 0.85, 0.55, 1.0),
            9: (0.20, 0.75, 0.30, 1.0), 12: (0.60, 0.85, 0.65, 1.0), 10: (0.75, 0.20, 0.75, 1.0), 13: (0.88, 0.65, 0.88, 1.0)
        }
        patches = []
        for tag in used_tags:
            if tag in TAG_COLORS and tag in tag_labels:
                patches.append(mpatches.Patch(color=TAG_COLORS[tag], label=tag_labels[tag]))
        if patches:
            ax.legend(handles=patches, loc='upper right', frameon=True, fontsize=8)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"✅ Saved mesh plot: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a 2D mesh (DOLFINx/Gmsh)")
    parser.add_argument("--mesh", required=False, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "motor.msh")),
                        help="Path to .msh file")
    parser.add_argument("--out", required=False, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "mesh_preview.png")),
                        help="Path to save output image (PNG). If omitted and --show is used, only display.")
    parser.add_argument("--show", action="store_true", help="Display interactively")
    parser.add_argument("--no-fill", action="store_true", help="Do not fill cells; edges only")
    parser.add_argument("--facets", dest="facets", action="store_true", help="Overlay facet segments")
    parser.add_argument("--realistic", action="store_true", help="Use realistic material colors and legend")
    args = parser.parse_args()

    plot_mesh(mesh_path=os.path.abspath(args.mesh),
              out_path=os.path.abspath(args.out) if args.out else None,
              show=bool(args.show),
              no_fill=bool(args.no_fill),
              draw_facets=bool(args.facets),
              realistic=bool(args.realistic))


if __name__ == "__main__":
    main()


