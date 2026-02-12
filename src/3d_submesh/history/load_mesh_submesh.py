"""Setup functions for 3D solver with submesh approach.

Order of execution (called from main_submesh.py):
  1. load_mesh_and_extract_submesh() - load parent mesh, create conductor submesh
  2. setup_materials() - assign sigma, nu, density on parent mesh
  3. setup_boundary_conditions_parent() - A=0 on exterior
  4. setup_boundary_conditions_submesh() - ground V on conductor
"""

import numpy as np
from dolfinx import fem, io
from dolfinx.mesh import locate_entities_boundary, create_submesh
from mpi4py import MPI
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "3d"))
from mesh_3D import model_parameters, surface_map
from load_mesh import CURRENT_MAP

# Domain tags (same as original)
AIR = (1,)
AIR_GAP = (2, 3)
ALUMINIUM = (4,)
ROTOR = (5,)
STATOR = (6,)
COILS = (7, 8, 9, 10, 11, 12)
MAGNETS = (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)


def conducting():
    """Conducting regions for σ-terms and the V-equation (include coils)."""
    return ROTOR + ALUMINIUM + MAGNETS + COILS


EXTERIOR_FACET_TAG = surface_map["Exterior"]


def load_mesh_and_extract_submesh(mesh_path):
    """
    Load mesh and extract conductor-only submesh.

    Returns:
    --------
    mesh_parent, mesh_conductor, cell_tags_parent, cell_tags_conductor,
    facet_tags_parent, conductor_cells, entity_map
    """
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
        from dolfinx.mesh import meshtags
        from entity_map_utils import entity_map_to_dict

        n_submesh_cells = mesh_conductor.topology.index_map(tdim).size_local
        submesh_cell_indices = np.arange(n_submesh_cells, dtype=np.int32)
        submesh_tags = np.empty(n_submesh_cells, dtype=np.int32)
        cell_to_tag_parent = {int(i): int(v) for i, v in zip(cell_tags_parent.indices, cell_tags_parent.values)}
        entity_dict = entity_map_to_dict(entity_map, n_submesh_cells, mesh_parent.comm)

        if len(entity_dict) > 0:
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
        conductor_cells,
        entity_map,
    )


def setup_materials(mesh, cell_tags, config):
    """Material parameters as DG0 Functions (sigma, nu, density) on parent mesh."""
    DG0 = fem.functionspace(mesh, ("DG", 0))
    sigma = fem.Function(DG0, name="sigma")
    nu = fem.Function(DG0, name="nu")
    density = fem.Function(DG0, name="density")

    mu_r = model_parameters["mu_r"]
    sigma_dict = model_parameters["sigma"]
    densities = model_parameters["densities"]

    marker_to_material = {
        1: "Air", 2: "AirGap", 3: "AirGap", 4: "Al", 5: "Rotor", 6: "Stator",
        **{m: "Cu" for m in COILS}, **{m: "PM" for m in MAGNETS},
    }

    # Small conductivity in air/air-gap regions to help solver convergence (optional)
    sigma_air_min = float(getattr(config, "sigma_air_min", 0.0))
    air_like_markers = AIR + AIR_GAP  # only bump sigma in these regions

    mu0 = config.mu0
    for marker, mat_name in marker_to_material.items():
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        sig = sigma_dict[mat_name]
        if sigma_air_min > 0 and sig == 0 and marker in air_like_markers:
            sig = sigma_air_min
        sigma.x.array[cells] = sig
        density.x.array[cells] = densities.get(mat_name, 0.0)
        nu.x.array[cells] = 1.0 / (mu0 * mu_r[mat_name])

    return sigma, nu, density


def setup_boundary_conditions_parent(mesh_parent, facet_tags_parent, A_space):
    """Strong Dirichlet BCs (A=0) on the exterior boundary of parent mesh."""
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


def setup_boundary_conditions_submesh(mesh_conductor, V_space, cell_tags_conductor, conductor_markers, config=None):
    """
    Set up boundary conditions for V on conductor submesh.
    - Current source: ground one DOF (first conductor cell).
    - Voltage source: ground one coil (e.g. coil 7), set V = V_amp*sin(omega*t+phase) on another coil (e.g. coil 8).
    Returns (bc_list, voltage_update_data). voltage_update_data is None for current drive, else (u_voltage, voltage_dofs, phase).
    """
    source_type = getattr(config, "source_type", "current") if config is not None else "current"

    # Ground: one dof from first conductor with cells (coil 7 for symmetry with voltage on coil 8)
    ground_marker = COILS[0] if source_type == "voltage" else conductor_markers[0]
    ground_dofs = np.array([], dtype=np.int32)
    for m in (ground_marker,) if source_type == "voltage" else conductor_markers:
        cells = cell_tags_conductor.find(m)
        if cells.size == 0:
            continue
        dofs = V_space.dofmap.cell_dofs(int(cells[0]))
        if len(dofs) == 0:
            continue
        ground_dofs = np.array([int(dofs[0])], dtype=np.int32)
        break
    u0 = fem.Function(V_space)
    u0.x.array[:] = 0.0
    bc_ground = fem.dirichletbc(u0, ground_dofs)

    if source_type != "voltage":
        return [bc_ground], None

    # Voltage drive: impose V on all dofs of one coil (e.g. coil 8)
    voltage_coil_marker = COILS[1]  # 8
    phase = CURRENT_MAP.get(voltage_coil_marker, {}).get("beta", 0.0)
    voltage_dofs_list = []
    for m in (voltage_coil_marker,):
        cells = cell_tags_conductor.find(m)
        for c in cells:
            dofs = V_space.dofmap.cell_dofs(int(c))
            voltage_dofs_list.extend(dofs.tolist())
    voltage_dofs = np.unique(np.array(voltage_dofs_list, dtype=np.int32))
    if voltage_dofs.size == 0:
        return [bc_ground], None
    u_voltage = fem.Function(V_space)
    u_voltage.x.array[:] = 0.0
    bc_voltage = fem.dirichletbc(u_voltage, voltage_dofs)
    return [bc_ground, bc_voltage], (u_voltage, voltage_dofs, phase)
