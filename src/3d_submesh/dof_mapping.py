"""DOF mapping between parent mesh and submesh for coupling assembly."""

import numpy as np
from dolfinx import fem
from entity_map_utils import entity_map_to_dict


class DOFMapper:
    """Maps DOFs between parent mesh (A-space) and submesh (V-space)."""

    def __init__(self, mesh_parent, mesh_conductor, A_space_parent, V_space_submesh,
                 entity_map, cell_tags_parent):
        self.mesh_parent = mesh_parent
        self.mesh_conductor = mesh_conductor
        self.A_space_parent = A_space_parent
        self.V_space_submesh = V_space_submesh
        self.entity_map = entity_map
        self.cell_tags_parent = cell_tags_parent
        self.comm = mesh_parent.comm
        self.rank = self.comm.rank
        self.tdim = mesh_parent.topology.dim
        self.A_dofmap = A_space_parent.dofmap
        self.V_dofmap = V_space_submesh.dofmap
        self._submesh_cell_to_parent_cell = {}
        self._parent_cell_to_submesh_cells = {}
        self._build_cell_mapping()

    def _build_cell_mapping(self):
        """Build mapping between submesh cells and parent cells."""
        n_submesh_cells = self.mesh_conductor.topology.index_map(self.tdim).size_local
        entity_dict = entity_map_to_dict(self.entity_map, n_submesh_cells, self.comm)
        if len(entity_dict) > 0:
            self._submesh_cell_to_parent_cell = entity_dict
            for submesh_cell, parent_cell in entity_dict.items():
                if parent_cell not in self._parent_cell_to_submesh_cells:
                    self._parent_cell_to_submesh_cells[parent_cell] = []
                self._parent_cell_to_submesh_cells[parent_cell].append(submesh_cell)
        else:
            if self.rank == 0:
                print("Warning: Could not extract entity map - mapping will be incomplete")

    def get_parent_cell_for_submesh_cell(self, submesh_cell):
        """Get parent cell index for a given submesh cell."""
        return self._submesh_cell_to_parent_cell.get(submesh_cell, -1)

    def get_submesh_cells_for_parent_cell(self, parent_cell):
        """Get all submesh cells that map to a given parent cell."""
        return self._parent_cell_to_submesh_cells.get(parent_cell, [])


def create_dof_mapper(mesh_parent, mesh_conductor, A_space_parent, V_space_submesh,
                     entity_map, cell_tags_parent):
    """Create a DOF mapper instance."""
    return DOFMapper(
        mesh_parent, mesh_conductor, A_space_parent, V_space_submesh,
        entity_map, cell_tags_parent
    )
