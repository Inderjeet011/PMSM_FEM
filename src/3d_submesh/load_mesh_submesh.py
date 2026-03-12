"""Setup functions for 3D solver with submesh approach."""

import numpy as np  # type: ignore
from dolfinx import fem, io  # type: ignore
from dolfinx.mesh import locate_entities_boundary, create_submesh  # type: ignore
from mpi4py import MPI  # type: ignore
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "3d"))
from mesh_3D import mesh_parameters, model_parameters, surface_map  # type: ignore

ROTOR = (5,)
COILS = (7, 8)
MAGNETS = (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
COIL_DRIVE = 7
COIL_GROUND = 8

# Single-phase voltage drive map.
VOLTAGE_MAP = [
    {"pos": 7, "neg": 8, "beta": 0.0},
]


def conducting():
    return COILS + MAGNETS + ROTOR


EXTERIOR_FACET_TAG = surface_map["Exterior"]


def load_mesh_and_extract_submesh(mesh_path):
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file {mesh_path} not found.")

    with io.XDMFFile(MPI.COMM_WORLD, str(mesh_path), "r") as xdmf:
        mesh_parent = xdmf.read_mesh()
        mesh_parent.topology.create_entities(mesh_parent.topology.dim - 1)
        cell_tags_parent = xdmf.read_meshtags(mesh_parent, name="cell_tags")
        facet_tags_parent = xdmf.read_meshtags(mesh_parent, name="facet_tags")

        # Try to read the DG0 field that marks lower straight coil halves.
        coil_lower_parent = None
        try:
            DG0_parent = fem.functionspace(mesh_parent, ("DG", 0))
            coil_lower_parent = xdmf.read_function(DG0_parent, name="CoilLowerHalves")
        except Exception:
            coil_lower_parent = None

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

    from dolfinx.mesh import meshtags  # type: ignore
    from entity_map_utils import entity_map_to_dict  # type: ignore

    n_submesh_cells = mesh_conductor.topology.index_map(tdim).size_local
    submesh_cell_indices = np.arange(n_submesh_cells, dtype=np.int32)
    submesh_tags = np.empty(n_submesh_cells, dtype=np.int32)
    cell_to_tag_parent = {int(i): int(v) for i, v in zip(cell_tags_parent.indices, cell_tags_parent.values)}
    entity_dict = entity_map_to_dict(entity_map, n_submesh_cells, mesh_parent.comm)

    # Optional: lower-half labels on the conductor submesh (from CoilLowerHalves DG0 field).
    coil_half_submesh = None
    if coil_lower_parent is not None:
        coil_half_submesh = np.zeros(n_submesh_cells, dtype=np.int32)
        parent_vals = coil_lower_parent.x.array
    else:
        parent_vals = None

    for i in range(n_submesh_cells):
        parent_cell = entity_dict.get(i, -1)
        submesh_tags[i] = cell_to_tag_parent.get(parent_cell, conductor_markers[0])
        if coil_half_submesh is not None and parent_vals is not None and 0 <= parent_cell < parent_vals.size:
            coil_half_submesh[i] = int(parent_vals[parent_cell])

    cell_tags_conductor = meshtags(mesh_conductor, tdim, submesh_cell_indices, submesh_tags)

    return (
        mesh_parent,
        mesh_conductor,
        cell_tags_parent,
        cell_tags_conductor,
        facet_tags_parent,
        entity_map,
        coil_half_submesh,
    )


def setup_materials(mesh, cell_tags, config):
    DG0 = fem.functionspace(mesh, ("DG", 0))
    sigma = fem.Function(DG0, name="sigma")
    nu = fem.Function(DG0, name="nu")

    mu_r = model_parameters["mu_r"]
    sigma_dict = model_parameters["sigma"]

    marker_to_material = {
        1: "Air", 2: "AirGap", 3: "AirGap", 4: "Al", 5: "Rotor", 6: "Stator",
        **{m: "Cu" for m in COILS}, **{m: "PM" for m in MAGNETS},
    }

    mu0 = config.mu0
    for marker, mat_name in marker_to_material.items():
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        sigma.x.array[cells] = sigma_dict[mat_name]
        nu.x.array[cells] = 1.0 / (mu0 * mu_r[mat_name])

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


def setup_boundary_conditions_submesh(
    mesh_conductor,
    V_space,
    cell_tags_conductor,
    conductor_markers,
    config,
    coil_half_submesh=None,  # kept for API compatibility, not used now
):
    """Return (bc_V_list, v_drive_funcs).

    Updated behaviour (for coil-rod experiments):

    - Treat EACH coil (7 and 8) as an isolated copper rod.
    - On the LOWER half of that rod: V = +V_amp
    - On the UPPER half of that rod: V = 0
    - Rotor/magnets get a single DOF pinned to 0 for gauge fixing.

    This removes the previous single-phase +/- V drive across the two coils.
    v_drive_funcs is returned empty; caller does not update voltages in time.
    """
    u0 = fem.Function(V_space)
    u0.x.array[:] = 0.0

    def half_dofs_for_marker(marker: int, which: str) -> np.ndarray:
        """
        Return DOFs in the LOWER or UPPER half of the straight coil leg
        for this marker, using only the conductor submesh geometry and
        cell tags.

        Strategy:
          1. Find all submesh cells with this coil marker (7 or 8).
          2. Compute each cell's center z-coordinate.
          3. Define z_mid = 0.5 * (z_min + z_max) for that coil.
          4. For 'lower': cells with center z <= z_mid (+tol).
             For 'upper': cells with center z >= z_mid (-tol).
          5. Collect all V-space DOFs on those cells.
        """
        tdim = mesh_conductor.topology.dim
        cells = cell_tags_conductor.find(marker)
        if cells.size == 0:
            return np.array([], dtype=np.int32)

        # Cell centers for these coil cells
        mesh_conductor.topology.create_connectivity(tdim, 0)
        c2v = mesh_conductor.topology.connectivity(tdim, 0)
        coords = mesh_conductor.geometry.x

        z_centers = []
        for c in cells:
            verts = c2v.links(int(c))
            if verts.size == 0:
                continue
            z_centers.append(float(coords[verts, 2].mean()))
        if not z_centers:
            return np.array([], dtype=np.int32)

        z_min = min(z_centers)
        z_max = max(z_centers)
        z_mid = 0.5 * (z_min + z_max)
        tol_z = 1e-8 * max(abs(z_max - z_min), 1.0)

        # Collect DOFs for cells in lower half (center z <= z_mid + tol)
        dm = V_space.dofmap
        dof_set = set()
        for c, zc in zip(cells, z_centers):
            if which == "lower":
                cond = (zc <= z_mid + tol_z)
            else:
                cond = (zc >= z_mid - tol_z)
            if cond:
                for d in dm.cell_dofs(int(c)):
                    dof_set.add(int(d))

        if not dof_set:
            return np.array([], dtype=np.int32)
        return np.unique(np.fromiter(dof_set, dtype=np.int32))

    bcs = []
    v_drive_funcs = []  # no time-varying drives in this mode
    driven_markers = set()

    # Apply V = V_amp on LOWER half, V = 0 on UPPER half of each coil.
    V_amp = float(getattr(config, "V_amp", 100.0))
    for marker in COILS:
        V_lower = fem.Function(V_space, name=f"V_lower_{marker}")
        V_upper = fem.Function(V_space, name=f"V_upper_{marker}")
        V_lower.x.array[:] = V_amp
        V_upper.x.array[:] = 0.0

        dofs_lower = half_dofs_for_marker(marker, "lower")
        dofs_upper = half_dofs_for_marker(marker, "upper")

        if dofs_lower.size > 0:
            bcs.append(fem.dirichletbc(V_lower, dofs_lower))
        if dofs_upper.size > 0:
            bcs.append(fem.dirichletbc(V_upper, dofs_upper))

        driven_markers.add(marker)

    # Gauge-fix any conductor region not covered by a phase voltage BC
    for m in conductor_markers:
        if m in driven_markers:
            continue
        cells = cell_tags_conductor.find(m)
        if cells.size > 0:
            fdofs = V_space.dofmap.cell_dofs(int(cells[0]))
            if fdofs.size > 0:
                bcs.append(fem.dirichletbc(u0, np.array([int(fdofs[0])], dtype=np.int32)))

    return bcs, v_drive_funcs
