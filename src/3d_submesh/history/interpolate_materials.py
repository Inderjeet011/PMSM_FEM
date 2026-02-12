"""Material property interpolation from parent mesh to submesh."""

import numpy as np
from dolfinx import fem
from entity_map_utils import entity_map_to_dict


def interpolate_sigma_to_submesh(mesh_parent, mesh_conductor, sigma_parent, entity_map, cell_tags_parent):
    """Interpolate sigma from parent mesh to conductor submesh using entity map."""
    DG0_submesh = fem.functionspace(mesh_conductor, ("DG", 0))
    sigma_submesh = fem.Function(DG0_submesh, name="sigma")
    tdim = mesh_conductor.topology.dim
    n_submesh_cells = mesh_conductor.topology.index_map(tdim).size_local
    sigma_submesh.x.array[:] = 0.0

    entity_dict = entity_map_to_dict(entity_map, n_submesh_cells, mesh_parent.comm)
    if len(entity_dict) > 0:
        for submesh_cell_idx in range(n_submesh_cells):
            parent_cell_idx = entity_dict.get(submesh_cell_idx, -1)
            if parent_cell_idx >= 0 and parent_cell_idx < sigma_parent.x.array.size:
                sigma_submesh.x.array[submesh_cell_idx] = sigma_parent.x.array[parent_cell_idx]
    else:
        sigma_submesh = interpolate_sigma_geometry_based(mesh_parent, mesh_conductor, sigma_parent)
        return sigma_submesh

    sigma_submesh.x.scatter_forward()
    return sigma_submesh


def interpolate_sigma_geometry_based(mesh_parent, mesh_conductor, sigma_parent):
    """Fallback: interpolate sigma using geometry-based cell location."""
    from dolfinx import geometry

    DG0_submesh = fem.functionspace(mesh_conductor, ("DG", 0))
    sigma_submesh = fem.Function(DG0_submesh, name="sigma")
    tdim = mesh_conductor.topology.dim
    mesh_conductor.topology.create_connectivity(tdim, 0)
    coords_submesh = mesh_conductor.geometry.x
    dofmap_submesh = mesh_conductor.geometry.dofmap
    centers_submesh = coords_submesh[dofmap_submesh].mean(axis=1)
    bb_tree = geometry.bb_tree(mesh_parent, tdim)
    n_submesh_cells = mesh_conductor.topology.index_map(tdim).size_local

    for i in range(n_submesh_cells):
        center = centers_submesh[i].reshape(1, -1)
        cell_candidates = geometry.compute_collisions_points(bb_tree, center)
        colliding_cells = geometry.compute_colliding_cells(mesh_parent, cell_candidates, center)
        if len(colliding_cells.links(0)) > 0:
            parent_cell = colliding_cells.links(0)[0]
            if parent_cell < sigma_parent.x.array.size:
                sigma_submesh.x.array[i] = sigma_parent.x.array[parent_cell]

    sigma_submesh.x.scatter_forward()
    return sigma_submesh


def interpolate_nu_to_submesh(mesh_parent, mesh_conductor, nu_parent, entity_map):
    """Interpolate nu from parent mesh to conductor submesh."""
    comm = mesh_parent.comm
    DG0_submesh = fem.functionspace(mesh_conductor, ("DG", 0))
    nu_submesh = fem.Function(DG0_submesh, name="nu")
    tdim = mesh_conductor.topology.dim
    n_submesh_cells = mesh_conductor.topology.index_map(tdim).size_local
    nu_submesh.x.array[:] = 0.0

    entity_dict = entity_map_to_dict(entity_map, n_submesh_cells, comm)
    if len(entity_dict) > 0:
        for submesh_cell_idx in range(n_submesh_cells):
            parent_cell_idx = entity_dict.get(submesh_cell_idx, -1)
            if parent_cell_idx >= 0 and parent_cell_idx < nu_parent.x.array.size:
                nu_submesh.x.array[submesh_cell_idx] = nu_parent.x.array[parent_cell_idx]
    else:
        nu_submesh = interpolate_sigma_geometry_based(mesh_parent, mesh_conductor, nu_parent)
        nu_submesh.name = "nu"
        nu_submesh.x.scatter_forward()
        return nu_submesh

    nu_submesh.x.scatter_forward()
    return nu_submesh
