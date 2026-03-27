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
TERMINAL_FACET_TAGS = {
    7: (701, 702),
    8: (801, 802),
    9: (901, 902),
    10: (1001, 1002),
    11: (1101, 1102),
    12: (1201, 1202),
}

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
    """Return voltage BCs using one representative DOF per exposed leg."""
    u0 = fem.Function(V_space)
    u0.x.array[:] = 0.0

    def _boundary_facets_for_coil_region(marker: int):
        tdim = mesh_conductor.topology.dim
        fdim = tdim - 1
        cells = cell_tags_conductor.find(marker)
        if cells.size == 0:
            return np.array([], dtype=np.int32)

        mesh_conductor.topology.create_connectivity(tdim, fdim)
        mesh_conductor.topology.create_connectivity(fdim, tdim)
        c2f = mesh_conductor.topology.connectivity(tdim, fdim)
        f2c = mesh_conductor.topology.connectivity(fdim, tdim)

        boundary_facets = []
        for c in cells:
            for f in c2f.links(int(c)):
                if len(f2c.links(int(f))) == 1:
                    boundary_facets.append(int(f))
        if not boundary_facets:
            return np.array([], dtype=np.int32)
        return np.unique(np.asarray(boundary_facets, dtype=np.int32))

    def _split_facet_components(facets: np.ndarray):
        if facets.size == 0:
            return []

        tdim = mesh_conductor.topology.dim
        fdim = tdim - 1
        mesh_conductor.topology.create_connectivity(fdim, 0)
        f2v = mesh_conductor.topology.connectivity(fdim, 0)

        vertex_to_facets = {}
        for f in facets:
            for v in f2v.links(int(f)):
                vertex_to_facets.setdefault(int(v), []).append(int(f))

        facet_set = {int(f) for f in facets}
        components = []
        visited = set()
        for start in facets:
            start_i = int(start)
            if start_i in visited:
                continue
            stack = [start_i]
            visited.add(start_i)
            comp = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for v in f2v.links(cur):
                    for neigh in vertex_to_facets.get(int(v), []):
                        if neigh in facet_set and neigh not in visited:
                            visited.add(neigh)
                            stack.append(neigh)
            components.append(np.asarray(comp, dtype=np.int32))
        return components

    def _facet_centroid_xyz(facet: int, f2v, coords):
        verts = f2v.links(int(facet))
        if verts.size == 0:
            return np.zeros(3, dtype=np.float64)
        return coords[verts].mean(axis=0)

    def _select_rectangular_terminal_pad(component: np.ndarray, f2v, coords):
        """Pick a small centered rectangular pad from one exposed leg cut face."""
        if component.size == 0:
            return component

        pad_fraction = float(getattr(config, "terminal_pad_fraction", 0.5))
        pad_fraction = min(max(pad_fraction, 0.15), 1.0)

        centroids = np.asarray(
            [_facet_centroid_xyz(int(f), f2v, coords) for f in component],
            dtype=np.float64,
        )
        component_center_xy = centroids[:, :2].mean(axis=0)
        verts = np.unique(np.concatenate([f2v.links(int(f)) for f in component]))
        pts_xy = coords[verts, :2]
        pts_xy_centered = pts_xy - component_center_xy

        if pts_xy.shape[0] >= 2:
            cov = np.cov(pts_xy_centered.T)
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            axes = evecs[:, order]
        else:
            axes = np.eye(2, dtype=np.float64)

        uv_verts = pts_xy_centered @ axes
        uv_centroids = (centroids[:, :2] - component_center_xy) @ axes

        spans = uv_verts.max(axis=0) - uv_verts.min(axis=0)
        spans = np.maximum(spans, 1e-12)
        half_widths = 0.5 * pad_fraction * spans

        mask = (
            (np.abs(uv_centroids[:, 0]) <= half_widths[0])
            & (np.abs(uv_centroids[:, 1]) <= half_widths[1])
        )
        selected = component[mask]

        if selected.size == 0:
            metric = (uv_centroids[:, 0] / half_widths[0]) ** 2 + (uv_centroids[:, 1] / half_widths[1]) ** 2
            selected = np.asarray([component[int(np.argmin(metric))]], dtype=np.int32)

        return selected.astype(np.int32)

    def terminal_facets_for_coil(marker: int):
        """
        Find the two terminal facet patches for a given coil.

        For the axial half-model, the terminals are the two disconnected open
        surfaces created by the cut plane. Only those facet patches receive
        Dirichlet data; the lateral coil surfaces remain floating.
        """
        tdim = mesh_conductor.topology.dim
        fdim = tdim - 1
        boundary_facets = _boundary_facets_for_coil_region(marker)
        if boundary_facets.size == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        mesh_conductor.topology.create_connectivity(fdim, tdim)
        mesh_conductor.topology.create_connectivity(fdim, 0)
        f2v = mesh_conductor.topology.connectivity(fdim, 0)
        coords = mesh_conductor.geometry.x

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

        if bool(getattr(config, "use_cut_face_terminals", False)) and getattr(config, "cut_plane_axis", "z") == "z":
            cut_facets = boundary_facets[z_mid <= (zmin + tol_z)]
            components = _split_facet_components(cut_facets)
            if len(components) >= 2:
                def component_key(comp: np.ndarray):
                    verts = np.unique(np.concatenate([f2v.links(int(f)) for f in comp]))
                    pts = coords[verts]
                    centroid = pts.mean(axis=0)
                    return (float(centroid[0]), float(centroid[1]), -int(comp.size))

                largest_two = sorted(components, key=lambda comp: int(comp.size), reverse=True)[:2]
                ordered = sorted(largest_two, key=component_key)
                drive_pad = _select_rectangular_terminal_pad(ordered[0], f2v, coords)
                neutral_pad = _select_rectangular_terminal_pad(ordered[1], f2v, coords)
                return drive_pad, neutral_pad

        facets_min = boundary_facets[z_mid <= (zmin + tol_z)]
        facets_max = boundary_facets[z_mid >= (zmax - tol_z)]
        return facets_min.astype(np.int32), facets_max.astype(np.int32)

    def _representative_leg_dof(marker: int, leg_facets: np.ndarray) -> np.ndarray:
        if leg_facets.size == 0:
            return np.array([], dtype=np.int32)
        tdim = mesh_conductor.topology.dim
        fdim = tdim - 1
        mesh_conductor.topology.create_connectivity(fdim, tdim)
        f2c = mesh_conductor.topology.connectivity(fdim, tdim)
        coil_cells = set(int(c) for c in cell_tags_conductor.find(marker))
        for f in leg_facets:
            for c in f2c.links(int(f)):
                ci = int(c)
                if ci in coil_cells:
                    dofs = V_space.dofmap.cell_dofs(ci)
                    if dofs.size > 0:
                        return np.array([int(dofs[0])], dtype=np.int32)
        return np.array([], dtype=np.int32)

    from dolfinx.mesh import meshtags  # type: ignore

    tdim = mesh_conductor.topology.dim
    fdim = tdim - 1
    terminal_facet_indices = []
    terminal_facet_values = []
    terminal_summary = []
    for marker in COILS:
        facets_drive, facets_neutral = terminal_facets_for_coil(marker)
        drive_tag, neutral_tag = TERMINAL_FACET_TAGS[marker]
        if facets_drive.size > 0:
            terminal_facet_indices.append(facets_drive)
            terminal_facet_values.append(np.full(facets_drive.size, drive_tag, dtype=np.int32))
        if facets_neutral.size > 0:
            terminal_facet_indices.append(facets_neutral)
            terminal_facet_values.append(np.full(facets_neutral.size, neutral_tag, dtype=np.int32))
        terminal_summary.append((marker, int(facets_drive.size), int(facets_neutral.size)))
        if facets_drive.size == 0 or facets_neutral.size == 0:
            raise RuntimeError(
                f"Could not identify both terminal end faces for coil {marker}: "
                f"drive facets={facets_drive.size}, neutral facets={facets_neutral.size}"
            )

    if terminal_facet_indices:
        facet_indices = np.concatenate(terminal_facet_indices).astype(np.int32)
        facet_values = np.concatenate(terminal_facet_values).astype(np.int32)
        order = np.argsort(facet_indices)
        terminal_mt = meshtags(mesh_conductor, fdim, facet_indices[order], facet_values[order])
    else:
        terminal_mt = meshtags(
            mesh_conductor,
            fdim,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    if mesh_conductor.comm.rank == 0:
        print("Tagged coil terminal facets (drive, neutral):")
        for marker, n_drive, n_neutral in terminal_summary:
            print(f"  coil {marker}: {n_drive}, {n_neutral}")

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
            drive_tag, neutral_tag = TERMINAL_FACET_TAGS[marker]
            facets_drive = terminal_mt.find(drive_tag)
            facets_neutral = terminal_mt.find(neutral_tag)
            dofs_drive = _representative_leg_dof(marker, facets_drive)
            dofs_neutral = _representative_leg_dof(marker, facets_neutral)
            if dofs_drive.size > 0:
                bcs.append(fem.dirichletbc(V_phase, dofs_drive))
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
