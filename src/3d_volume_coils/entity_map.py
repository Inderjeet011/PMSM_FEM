"""
Helpers for DOLFINx ``EntityMap`` between conductor submesh and parent mesh.

Used when projecting submesh fields or copying parent-cell data (e.g. \\(\\sigma\\))
to conductor cells. ``get_entity_map`` returns the submesh-to-parent cell index
array; ``entity_map_to_dict`` builds a dense Python map for small loops.
"""

import numpy as np
import numpy.typing as npt
from dolfinx import mesh


def get_entity_map(entity_map: mesh.EntityMap, inverse: bool = False) -> npt.NDArray[np.int32]:
    """Get an entity map from the sub-topology to the topology."""
    sub_top = entity_map.sub_topology
    assert isinstance(sub_top, mesh.Topology)
    sub_map = sub_top.index_map(entity_map.dim)
    indices = np.arange(sub_map.size_local + sub_map.num_ghosts, dtype=np.int32)
    return entity_map.sub_topology_to_topology(indices, inverse=inverse)


def entity_map_to_dict(entity_map, n_submesh_cells):
    """Convert EntityMap to a dictionary mapping submesh cell -> parent cell."""
    mapping = {}
    parent_cells_array = get_entity_map(entity_map, inverse=False)
    for submesh_cell in range(min(n_submesh_cells, len(parent_cells_array))):
        parent_cell = int(parent_cells_array[submesh_cell])
        mapping[submesh_cell] = parent_cell
    return mapping