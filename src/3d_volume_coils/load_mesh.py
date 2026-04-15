"""
Mesh loading and problem setup for ``3d_volume_coils``.

Reads ``mesh.xdmf``, extracts the conductor submesh and ``EntityMap``, maps
parent cell tags onto the submesh, and defines volume/facet markers (air, coils,
PMs, exterior, …). Provides ``setup_materials`` (\\(\\sigma\\), \\(\\nu\\)) and
boundary conditions: strong ``A = 0`` on the outer air shell and fixed
potentials / constraints on the conductor submesh as used by ``main.py``.
"""

import numpy as np  # type: ignore
from dolfinx import fem, io  # type: ignore
from dolfinx.mesh import locate_entities_boundary, create_submesh  # type: ignore
from mpi4py import MPI  # type: ignore
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mesh import model_parameters, surface_map  # type: ignore

AIR = (1,)
AIR_GAP = (2, 3)
# Shaft + rotor iron: single tag in mesh (see ``mesh._domain_map_three``)
ROTOR = (5,)
STATOR = (6,)
COILS = (7, 8, 9, 10, 11, 12)
MAGNETS = (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
CURRENT_MAP = {
    7: {"alpha": 1, "beta": 0.0},
    8: {"alpha": -1, "beta": 2.0 * np.pi / 3.0},
    9: {"alpha": 1, "beta": 4.0 * np.pi / 3.0},
    10: {"alpha": -1, "beta": 0.0},
    11: {"alpha": 1, "beta": 2.0 * np.pi / 3.0},
    12: {"alpha": -1, "beta": 4.0 * np.pi / 3.0},
}


def conducting():
    return COILS + MAGNETS + ROTOR


def omega_r():
    """Rotor assembly (inner stack and annulus; one mesh tag)."""
    return ROTOR


def omega_s():
    return STATOR


def omega_pm():
    return MAGNETS


def omega_c():
    return COILS


def omega_rpm():
    return ROTOR + MAGNETS


EXTERIOR_FACET_TAG = surface_map["Exterior"]


def load_mesh_and_extract_submesh(mesh_path):
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file {mesh_path} not found.")

    with io.XDMFFile(MPI.COMM_WORLD, str(mesh_path), "r") as xdmf:
        mesh_parent = xdmf.read_mesh()
        mesh_parent.topology.create_entities(mesh_parent.topology.dim - 1)
        cell_tags_parent = xdmf.read_meshtags(mesh_parent, name="cell_tags")
        facet_tags_parent = xdmf.read_meshtags(mesh_parent, name="facet_tags")

    conductor_markers = conducting()
    conductor_cells = np.array([], dtype=np.int32)
    for marker in conductor_markers:
        cells = cell_tags_parent.find(marker)
        if cells.size > 0:
            conductor_cells = np.concatenate([conductor_cells, cells])
    conductor_cells = np.unique(conductor_cells)

    tdim = mesh_parent.topology.dim
    result = create_submesh(mesh_parent, tdim, conductor_cells)
    mesh_conductor = result[0]
    entity_map = result[1]

    cell_tags_conductor = None
    if cell_tags_parent is not None:
        from dolfinx.mesh import meshtags  # type: ignore
        from entity_map import entity_map_to_dict  # type: ignore

        n_submesh_cells = mesh_conductor.topology.index_map(tdim).size_local
        submesh_cell_indices = np.arange(n_submesh_cells, dtype=np.int32)
        submesh_tags = np.empty(n_submesh_cells, dtype=np.int32)
        cell_to_tag_parent = {int(i): int(v) for i, v in zip(cell_tags_parent.indices, cell_tags_parent.values)}
        entity_dict = entity_map_to_dict(entity_map, n_submesh_cells)
        for i in range(n_submesh_cells):
            parent_cell = entity_dict.get(i, -1)
            submesh_tags[i] = cell_to_tag_parent.get(parent_cell, conductor_markers[0])
        cell_tags_conductor = meshtags(mesh_conductor, tdim, submesh_cell_indices, submesh_tags)

    return (
        mesh_parent,
        mesh_conductor,
        cell_tags_parent,
        cell_tags_conductor,
        facet_tags_parent,
        entity_map,
    )


def setup_materials(mesh, cell_tags, config):
    DG0 = fem.functionspace(mesh, ("DG", 0))
    sigma = fem.Function(DG0, name="sigma")
    nu = fem.Function(DG0, name="nu")

    mu_r = model_parameters["mu_r"]
    sigma_dict = model_parameters["sigma"]

    # Tag 4: legacy shaft-only cells; new meshes use tag 5 for shaft + rotor iron
    marker_to_material = {
        1: "Air", 2: "AirGap", 3: "AirGap", 4: "Rotor", 5: "Rotor", 6: "Stator",
        **{m: "Cu" for m in COILS}, **{m: "PM" for m in MAGNETS},
    }

    mu0 = config.mu0
    for marker, mat_name in marker_to_material.items():
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        sigma.x.array[cells] = sigma_dict[mat_name]
        nu.x.array[cells] = 1.0 / (mu0 * mu_r[mat_name])

    sigma_cu_override = getattr(config, "sigma_cu_override", None)
    if sigma_cu_override is not None and float(sigma_cu_override) > 0:
        sig_cu = float(sigma_cu_override)
        for m in COILS:
            cells = cell_tags.find(m)
            sigma.x.array[cells] = sig_cu

    return sigma, nu


def setup_boundary_conditions_parent(mesh_parent, facet_tags_parent, A_space):
    tdim = mesh_parent.topology.dim
    mesh_parent.topology.create_connectivity(tdim - 1, tdim)
    exterior_facets = (
        facet_tags_parent.find(EXTERIOR_FACET_TAG)
        if facet_tags_parent is not None
        else locate_entities_boundary(mesh_parent, tdim - 1, lambda x: np.full(x.shape[1], True))
    )
    u0 = fem.Function(A_space)
    u0.x.array[:] = 0.0
    dofs = fem.locate_dofs_topological(A_space, tdim - 1, exterior_facets)
    return fem.dirichletbc(u0, dofs)


def setup_boundary_conditions_submesh(mesh_conductor, V_space, cell_tags_conductor, conductor_markers):
    u0 = fem.Function(V_space)
    u0.x.array[:] = 0.0
    ground_dofs_list = []
    for m in conductor_markers:
        cells = cell_tags_conductor.find(m)
        if cells.size > 0:
            dofs = V_space.dofmap.cell_dofs(int(cells[0]))
            if len(dofs) > 0:
                ground_dofs_list.append(int(dofs[0]))
    ground_dofs = np.unique(np.array(ground_dofs_list, dtype=np.int32))
    return [fem.dirichletbc(u0, ground_dofs)]