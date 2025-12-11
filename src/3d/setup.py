"""
Setup module for 3D A-V solver.

This module handles all initialization tasks:
- Configuration
- Domain metadata
- Mesh loading
- Material properties
- Boundary conditions
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dolfinx import fem, io, mesh as dmesh
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI

from mesh_3D import mesh_parameters, model_parameters, surface_map


# ============================================================================
# Domain metadata
# ============================================================================

class DomainTags3D:
    """Physical cell markers present in the 3D mesh."""

    AIR: tuple[int, ...] = (1,)
    AIR_GAP: tuple[int, ...] = (2, 3)
    ALUMINIUM: tuple[int, ...] = (4,)
    ROTOR: tuple[int, ...] = (5,)
    STATOR: tuple[int, ...] = (6,)
    COILS: tuple[int, ...] = (7, 8, 9, 10, 11, 12)
    MAGNETS: tuple[int, ...] = (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)

    @classmethod
    def conducting(cls) -> tuple[int, ...]:
        """Return tags with non-zero conductivity."""
        return cls.ROTOR + cls.ALUMINIUM + cls.MAGNETS

    @classmethod
    def all_cells(cls) -> tuple[int, ...]:
        """All cell markers present in the mesh."""
        return (
            cls.AIR
            + cls.AIR_GAP
            + cls.ALUMINIUM
            + cls.ROTOR
            + cls.STATOR
            + cls.COILS
            + cls.MAGNETS
        )


# Three-phase current mapping for coils
CURRENT_MAP = {
    7: {"alpha": 1.0, "beta": 0.0},
    8: {"alpha": -1.0, "beta": 2 * np.pi / 3},
    9: {"alpha": 1.0, "beta": 4 * np.pi / 3},
    10: {"alpha": -1.0, "beta": 0.0},
    11: {"alpha": 1.0, "beta": 2 * np.pi / 3},
    12: {"alpha": -1.0, "beta": 4 * np.pi / 3},
}

EXTERIOR_FACET_TAG = surface_map["Exterior"]


# ============================================================================
# Configuration
# ============================================================================

@dataclass(slots=True)
class SimulationConfig3D:
    """User-configurable solver parameters."""

    pole_pairs: int = 2
    drive_frequency: float = model_parameters["freq"]
    steps_per_period: int = 20  # Number of time steps per electrical period
    dt: float = 0.0
    num_steps: int = 0
    degree_A: int = 1
    degree_V: int = 1
    coil_current_peak: float = model_parameters["J"]
    mu0: float = model_parameters["mu_0"]
    mesh_path: Path = Path(__file__).parents[2] / "meshes" / "3d" / "pmesh3D_test1.xdmf"
    results_path: Path = Path(__file__).parents[2] / "results" / "3d" / "av_solver.xdmf"
    diagnostics_path: Path = Path(__file__).parents[2] / "results" / "3d" / "av_solver_diagnostics.csv"
    write_results: bool = True
    magnet_remanence: float = 1.2  # Tesla (NdFeB grade)

    def __post_init__(self) -> None:
        self.mesh_path = Path(self.mesh_path)
        self.results_path = Path(self.results_path)
        self.diagnostics_path = Path(self.diagnostics_path)
        period = 1.0 / self.drive_frequency
        self.dt = period / self.steps_per_period
        self.num_steps = self.steps_per_period

    @property
    def omega_e(self) -> float:
        """Electrical angular frequency."""
        return 2 * np.pi * self.drive_frequency

    @property
    def omega_m(self) -> float:
        """Mechanical angular frequency."""
        return self.omega_e / max(self.pole_pairs, 1)


# ============================================================================
# Mesh utilities
# ============================================================================

def load_mesh(mesh_path):
    """Read XDMF mesh + tags exported by mesh_3D.py."""
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Mesh file {mesh_path} not found. Generate it via mesh_3D.py first."
        )

    with io.XDMFFile(MPI.COMM_WORLD, str(mesh_path), "r") as xdmf:
        mesh = xdmf.read_mesh()
        cell_tags = xdmf.read_meshtags(mesh, name="Cell_markers")
        mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
        facet_tags = xdmf.read_meshtags(mesh, name="Facet_markers")

    if mesh.comm.rank == 0:
        print(
            f"📦 Mesh loaded: {mesh.topology.index_map(mesh.topology.dim).size_global} cells"
        )
    
    return mesh, cell_tags, facet_tags


def maybe_retag_cells(mesh, cell_tags):
    """Reconstruct cell markers if the mesh lacks required tags."""
    if cell_tags is None:
        raise RuntimeError("Mesh must provide cell tags from gmsh.")

    current_tags = set(np.unique(cell_tags.values))
    required_tags = {
        DomainTags3D.ROTOR[0],
        DomainTags3D.ALUMINIUM[0],
        DomainTags3D.STATOR[0],
        DomainTags3D.AIR[0],
        DomainTags3D.AIR_GAP[0],
        DomainTags3D.AIR_GAP[1],
    }
    required_tags.update(DomainTags3D.COILS)
    required_tags.update(DomainTags3D.MAGNETS)

    if required_tags.issubset(current_tags):
        return cell_tags

    if mesh.comm.rank == 0:
        print("⚠️  Retagging cells based on geometry (gmsh tags incomplete).")

    coords = mesh.geometry.x
    dofmap = mesh.geometry.dofmap
    centers = coords[dofmap].mean(axis=1)
    radii = np.linalg.norm(centers[:, :2], axis=1)
    angles = np.mod(np.arctan2(centers[:, 1], centers[:, 0]), 2 * np.pi)

    r1 = mesh_parameters["r1"]
    r2 = mesh_parameters["r2"]
    r3 = mesh_parameters["r3"]
    r4 = mesh_parameters["r4"]
    r5 = mesh_parameters["r5"]
    r6 = mesh_parameters["r6"]
    r7 = mesh_parameters["r7"]
    r_mid = 0.5 * (r2 + r3)

    coil_spacing = (np.pi / 4) + (np.pi / 4) / 3
    coil_centers = np.asarray([i * coil_spacing for i in range(len(DomainTags3D.COILS))])
    pm_spacing = (np.pi / 6) + (np.pi / 30)
    pm_centers = np.asarray([i * pm_spacing for i in range(len(DomainTags3D.MAGNETS))])

    coil_half = np.pi / 8 + np.deg2rad(2.0)
    pm_half = np.pi / 12 + np.deg2rad(2.0)
    radial_tol = 5e-4

    def _nearest(theta: float, centers: np.ndarray) -> tuple[int, float]:
        diffs = np.arctan2(np.sin(theta - centers), np.cos(theta - centers))
        idx = int(np.argmin(np.abs(diffs)))
        return idx, float(abs(diffs[idx]))

    new_tags = np.empty_like(cell_tags.values)
    for cell in range(len(new_tags)):
        r = radii[cell]
        theta = angles[cell]
        tag = DomainTags3D.AIR[0]

        if r <= r1 + radial_tol:
            tag = DomainTags3D.ROTOR[0]
        elif r <= r6 - radial_tol:
            tag = DomainTags3D.ALUMINIUM[0]
        elif r <= r7 + radial_tol:
            idx, delta = _nearest(theta, pm_centers)
            if delta <= pm_half:
                tag = DomainTags3D.MAGNETS[idx]
            else:
                tag = DomainTags3D.ALUMINIUM[0]
        elif r <= r2 - radial_tol:
            tag = DomainTags3D.ALUMINIUM[0]
        elif r <= r_mid + radial_tol:
            tag = DomainTags3D.AIR_GAP[0]
        elif r <= r3 + radial_tol:
            tag = DomainTags3D.AIR_GAP[1]
        elif r <= r4 + radial_tol:
            idx, delta = _nearest(theta, coil_centers)
            if delta <= coil_half:
                tag = DomainTags3D.COILS[idx]
            else:
                tag = DomainTags3D.AIR[0]
        elif r <= r5 + radial_tol:
            tag = DomainTags3D.STATOR[0]
        else:
            tag = DomainTags3D.AIR[0]

        new_tags[cell] = tag

    cell_indices = np.arange(len(new_tags), dtype=np.int32)
    new_cell_tags = dmesh.meshtags(mesh, mesh.topology.dim, cell_indices, new_tags)
    if mesh.comm.rank == 0:
        print(f"   Retagged mesh. Unique cell tags: {sorted(set(new_tags.tolist()))}")
    
    return new_cell_tags


# ============================================================================
# Material properties
# ============================================================================

def setup_materials(mesh, cell_tags, config):
    """Populate piecewise-constant fields (σ, ν, ρ)."""
    DG0 = fem.functionspace(mesh, ("DG", 0))
    sigma = fem.Function(DG0, name="sigma")
    nu = fem.Function(DG0, name="nu")
    density = fem.Function(DG0, name="density")

    mu_r = model_parameters["mu_r"]
    sigma_dict = model_parameters["sigma"]
    densities = model_parameters["densities"]

    marker_to_material = {
        1: "Air",
        2: "AirGap",
        3: "AirGap",
        4: "Al",
        5: "Rotor",
        6: "Stator",
    }

    for marker in DomainTags3D.COILS:
        marker_to_material[marker] = "Cu"
    for marker in DomainTags3D.MAGNETS:
        marker_to_material[marker] = "PM"

    mu0 = config.mu0
    for marker, mat_name in marker_to_material.items():
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        sigma.x.array[cells] = sigma_dict[mat_name]
        density.x.array[cells] = densities.get(mat_name, 0.0)
        mu_rel = mu_r[mat_name]
        nu.x.array[cells] = 1.0 / (mu0 * mu_rel)

    if mesh.comm.rank == 0:
        print("✅ Material fields assigned (σ, ν, ρ)")

    return sigma, nu, density


# ============================================================================
# Boundary conditions
# ============================================================================

def setup_boundary_conditions(mesh, facet_tags, A_space, V_space):
    """Impose A = 0 and V = 0 on the exterior surface."""
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)

    if facet_tags is not None:
        exterior_facets = facet_tags.find(EXTERIOR_FACET_TAG)
    else:
        exterior_facets = locate_entities_boundary(
            mesh, tdim - 1, lambda _x: np.full(_x.shape[1], True)
        )

    zero_vector = fem.Function(A_space)
    zero_vector.x.array[:] = 0.0
    bdofs_A = fem.locate_dofs_topological(
        A_space, entity_dim=tdim - 1, entities=exterior_facets
    )
    bc_A = fem.dirichletbc(zero_vector, bdofs_A)

    zero_scalar = fem.Function(V_space)
    zero_scalar.x.array[:] = 0.0
    bdofs_V = fem.locate_dofs_topological(
        V_space, entity_dim=tdim - 1, entities=exterior_facets
    )
    bc_V = fem.dirichletbc(zero_scalar, bdofs_V)

    if mesh.comm.rank == 0:
        print(
            f"🔒 Boundary DOFs -> A: {bdofs_A.size}, V: {bdofs_V.size} (tag={EXTERIOR_FACET_TAG})"
        )

    block_bcs = [[bc_A], [bc_V]]
    return bc_A, bc_V, block_bcs

