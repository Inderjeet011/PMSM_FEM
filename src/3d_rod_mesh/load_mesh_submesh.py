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
# Six copper coils (volume tags 7–12)
COILS = (7, 8, 9, 10, 11, 12)
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

    3‑phase voltage‑driven coils, with each copper rod acting as its own
    bar conductor:

    - There are six coils: markers (7, 8, 9, 10, 11, 12).
    - We group them into three electrical phases:
        Phase A: 7 and 10   (β_A = 0)
        Phase B: 8 and 11   (β_B = 2π/3)
        Phase C: 9 and 12   (β_C = 4π/3)
    - For each coil in a phase (bar-conductor terminals):
        * One axial end face (z ≈ z_min): V = V_amp * sin(ω_e t + β_phase)
        * The opposite axial end face (z ≈ z_max): V = 0   (neutral)
    - Rotor/magnets get a single DOF pinned to 0 for gauge fixing.

    The function returns:
        bcs           : list of DirichletBC objects
        v_drive_funcs : list of dicts, one per phase:
                        {"func": fem.Function, "beta": float}
    The caller (main_submesh) updates each phase["func"] in time.
    """
    u0 = fem.Function(V_space)
    u0.x.array[:] = 0.0

    def terminal_facets_for_coil(marker: int):
        """
        Find the two axial end-face facet sets for a given coil marker.

        We restrict to *boundary facets* of the coil region and pick facets
        near the minimum and maximum facet-midpoint z. This avoids clamping
        an entire half-volume and produces a smooth end-to-end potential
        gradient like a copper-rod test.
        """
        tdim = mesh_conductor.topology.dim
        fdim = tdim - 1
        cells = cell_tags_conductor.find(marker)
        if cells.size == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        # Ensure needed connectivity.
        mesh_conductor.topology.create_connectivity(tdim, fdim)
        mesh_conductor.topology.create_connectivity(fdim, tdim)
        mesh_conductor.topology.create_connectivity(fdim, 0)

        c2f = mesh_conductor.topology.connectivity(tdim, fdim)
        f2c = mesh_conductor.topology.connectivity(fdim, tdim)
        f2v = mesh_conductor.topology.connectivity(fdim, 0)
        coords = mesh_conductor.geometry.x

        # Boundary facets that belong to this coil region
        boundary_facets = []
        for c in cells:
            for f in c2f.links(int(c)):
                # A boundary facet has only one adjacent cell.
                if len(f2c.links(int(f))) == 1:
                    boundary_facets.append(int(f))
        if not boundary_facets:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        boundary_facets = np.unique(np.asarray(boundary_facets, dtype=np.int32))

        # Facet midpoint z
        z_mid = np.empty(boundary_facets.size, dtype=np.float64)
        for i, f in enumerate(boundary_facets):
            verts = f2v.links(int(f))
            if verts.size == 0:
                z_mid[i] = 0.0
            else:
                z_mid[i] = float(coords[verts, 2].mean())

        zmin = float(z_mid.min())
        zmax = float(z_mid.max())
        zspan = max(abs(zmax - zmin), 1.0)
        tol_z = 1e-6 * zspan

        facets_min = boundary_facets[z_mid <= (zmin + tol_z)]
        facets_max = boundary_facets[z_mid >= (zmax - tol_z)]
        return facets_min.astype(np.int32), facets_max.astype(np.int32)

    bcs = []
    v_drive_funcs = []
    driven_markers = set()

    # Phase grouping: 3 phases, 2 coils per phase.
    two_pi_over_3 = 2.0 * np.pi / 3.0
    phase_definitions = [
        {"coils": (7, 10), "beta": 0.0},
        {"coils": (8, 11), "beta": two_pi_over_3},
        {"coils": (9, 12), "beta": 2.0 * two_pi_over_3},
    ]

    # Neutral potential (0 V) used on upper halves.
    V_neutral = fem.Function(V_space)
    V_neutral.x.array[:] = 0.0

    for phase in phase_definitions:
        V_phase = fem.Function(V_space, name=f"V_phase_{phase['coils'][0]}_{phase['coils'][1]}")
        V_phase.x.array[:] = 0.0

        for marker in phase["coils"]:
            facets_drive, facets_neutral = terminal_facets_for_coil(marker)
            if facets_drive.size > 0:
                dofs_drive = fem.locate_dofs_topological(V_space, mesh_conductor.topology.dim - 1, facets_drive)
                if dofs_drive.size > 0:
                    bcs.append(fem.dirichletbc(V_phase, dofs_drive))
            if facets_neutral.size > 0:
                dofs_neutral = fem.locate_dofs_topological(V_space, mesh_conductor.topology.dim - 1, facets_neutral)
                if dofs_neutral.size > 0:
                    bcs.append(fem.dirichletbc(V_neutral, dofs_neutral))

            driven_markers.add(marker)

        v_drive_funcs.append({"func": V_phase, "beta": phase["beta"]})

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
