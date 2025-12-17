"""Setup functions for 3D solver."""

from pathlib import Path
import numpy as np
from dolfinx import fem, io, mesh as dmesh
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from mesh_3D import mesh_parameters, model_parameters, surface_map


# Domain tags
AIR = (1,)
AIR_GAP = (2, 3)
ALUMINIUM = (4,)
ROTOR = (5,)
STATOR = (6,)
COILS = (7, 8, 9, 10, 11, 12)
MAGNETS = (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)

def conducting():
    return ROTOR + ALUMINIUM + MAGNETS

# Current mapping
CURRENT_MAP = {
    7: {"alpha": 1.0, "beta": 0.0},
    8: {"alpha": -1.0, "beta": 2 * np.pi / 3},
    9: {"alpha": 1.0, "beta": 4 * np.pi / 3},
    10: {"alpha": -1.0, "beta": 0.0},
    11: {"alpha": 1.0, "beta": 2 * np.pi / 3},
    12: {"alpha": -1.0, "beta": 4 * np.pi / 3},
}

EXTERIOR_FACET_TAG = surface_map["Exterior"]


class SimulationConfig3D:
    """Solver configuration."""
    
    def __init__(self):
        self.pole_pairs = 2
        self.drive_frequency = model_parameters["freq"]
        self.steps_per_period = 20
        period = 1.0 / self.drive_frequency
        self.dt = period / self.steps_per_period
        # Run enough steps to cover at least 16ms (for visualization at 4ms, 8ms, 12ms, 16ms)
        # With dt ≈ 0.833ms, we need at least 20 steps to reach 16ms
        self.num_steps = 20  # This gives ~16.67ms total time, covering all requested timesteps
        self.degree_A = 1
        self.degree_V = 1
        self.coil_current_peak = model_parameters["J"]
        self.mu0 = model_parameters["mu_0"]
        self.mesh_path = Path(__file__).parents[2] / "meshes" / "3d" / "pmesh3D_ipm.xdmf"
        self.results_path = Path(__file__).parents[2] / "results" / "3d" / "av_solver.xdmf"
        self.diagnostics_path = Path(__file__).parents[2] / "results" / "3d" / "av_solver_diagnostics.csv"
        self.write_results = True
        self.magnet_remanence = 1.2
    
    @property
    def omega_e(self):
        return 2 * np.pi * self.drive_frequency
    
    @property
    def omega_m(self):
        return self.omega_e / max(self.pole_pairs, 1)


# Keep DomainTags3D for backward compatibility
class DomainTags3D:
    AIR = AIR
    AIR_GAP = AIR_GAP
    ALUMINIUM = ALUMINIUM
    ROTOR = ROTOR
    STATOR = STATOR
    COILS = COILS
    MAGNETS = MAGNETS
    
    @classmethod
    def conducting(cls):
        return conducting()


def load_mesh(mesh_path):
    """Load mesh from XDMF file."""
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file {mesh_path} not found.")
    
    with io.XDMFFile(MPI.COMM_WORLD, str(mesh_path), "r") as xdmf:
        mesh = xdmf.read_mesh()
        # Try different possible names for cell tags
        cell_tags = None
        for name in ["Cell_markers", "mesh_tags"]:
            try:
                cell_tags = xdmf.read_meshtags(mesh, name=name)
                break
            except:
                continue
        
        mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
        
        # Try different possible names for facet tags
        facet_tags = None
        for name in ["Facet_markers", "facet_tags"]:
            try:
                facet_tags = xdmf.read_meshtags(mesh, name=name)
                break
            except:
                continue
    
    if mesh.comm.rank == 0:
        print(f"Mesh loaded: {mesh.topology.index_map(mesh.topology.dim).size_global} cells")
    
    return mesh, cell_tags, facet_tags


def maybe_retag_cells(mesh, cell_tags):
    """Retag cells if needed."""
    if cell_tags is None:
        raise RuntimeError("Mesh must provide cell tags.")
    
    current_tags = set(np.unique(cell_tags.values))
    required_tags = {ROTOR[0], ALUMINIUM[0], STATOR[0], AIR[0], AIR_GAP[0], AIR_GAP[1]}
    required_tags.update(COILS)
    required_tags.update(MAGNETS)
    
    if required_tags.issubset(current_tags):
        return cell_tags
    
    if mesh.comm.rank == 0:
        print("Retagging cells...")
    
    coords = mesh.geometry.x
    dofmap = mesh.geometry.dofmap
    centers = coords[dofmap].mean(axis=1)
    radii = np.linalg.norm(centers[:, :2], axis=1)
    angles = np.mod(np.arctan2(centers[:, 1], centers[:, 0]), 2 * np.pi)
    
    r1, r2, r3, r4, r5, r6, r7 = [mesh_parameters[f"r{i}"] for i in range(1, 8)]
    r_mid = 0.5 * (r2 + r3)
    
    coil_spacing = (np.pi / 4) + (np.pi / 4) / 3
    coil_centers = np.asarray([i * coil_spacing for i in range(len(COILS))])
    pm_spacing = (np.pi / 6) + (np.pi / 30)
    pm_centers = np.asarray([i * pm_spacing for i in range(len(MAGNETS))])
    
    coil_half = np.pi / 8 + np.deg2rad(2.0)
    pm_half = np.pi / 12 + np.deg2rad(2.0)
    radial_tol = 5e-4
    
    def nearest(theta, centers):
        diffs = np.arctan2(np.sin(theta - centers), np.cos(theta - centers))
        idx = int(np.argmin(np.abs(diffs)))
        return idx, float(abs(diffs[idx]))
    
    new_tags = np.empty_like(cell_tags.values)
    for cell in range(len(new_tags)):
        r = radii[cell]
        theta = angles[cell]
        tag = AIR[0]
        
        if r <= r1 + radial_tol:
            tag = ROTOR[0]
        elif r <= r6 - radial_tol:
            tag = ALUMINIUM[0]
        elif r <= r7 + radial_tol:
            idx, delta = nearest(theta, pm_centers)
            tag = MAGNETS[idx] if delta <= pm_half else ALUMINIUM[0]
        elif r <= r2 - radial_tol:
            tag = ALUMINIUM[0]
        elif r <= r_mid + radial_tol:
            tag = AIR_GAP[0]
        elif r <= r3 + radial_tol:
            tag = AIR_GAP[1]
        elif r <= r4 + radial_tol:
            idx, delta = nearest(theta, coil_centers)
            tag = COILS[idx] if delta <= coil_half else AIR[0]
        elif r <= r5 + radial_tol:
            tag = STATOR[0]
        else:
            tag = AIR[0]
        
        new_tags[cell] = tag
    
    cell_indices = np.arange(len(new_tags), dtype=np.int32)
    new_cell_tags = dmesh.meshtags(mesh, mesh.topology.dim, cell_indices, new_tags)
    
    if mesh.comm.rank == 0:
        print(f"Retagged mesh. Unique tags: {sorted(set(new_tags.tolist()))}")
    
    return new_cell_tags


def setup_materials(mesh, cell_tags, config):
    """Setup material properties."""
    DG0 = fem.functionspace(mesh, ("DG", 0))
    sigma = fem.Function(DG0, name="sigma")
    nu = fem.Function(DG0, name="nu")
    density = fem.Function(DG0, name="density")
    
    mu_r = model_parameters["mu_r"]
    sigma_dict = model_parameters["sigma"]
    densities = model_parameters["densities"]
    
    marker_to_material = {
        1: "Air", 2: "AirGap", 3: "AirGap", 4: "Al", 5: "Rotor", 6: "Stator",
    }
    
    for marker in COILS:
        marker_to_material[marker] = "Cu"
    for marker in MAGNETS:
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
        print("Materials assigned")
    
    return sigma, nu, density


def setup_boundary_conditions(mesh, facet_tags, A_space, V_space):
    """Setup boundary conditions.
    
    For AMS compatibility, we use weak boundary conditions on A (penalty term)
    and keep strong Dirichlet BC on V (scalar potential).
    """
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    
    if facet_tags is not None:
        exterior_facets = facet_tags.find(EXTERIOR_FACET_TAG)
    else:
        exterior_facets = locate_entities_boundary(
            mesh, tdim - 1, lambda _x: np.full(_x.shape[1], True)
        )
    
    # No strong Dirichlet BC on A_space (weak penalty term will be used instead)
    bc_A = None
    
    # Keep strong Dirichlet BC on V_space (scalar potential)
    zero_scalar = fem.Function(V_space)
    zero_scalar.x.array[:] = 0.0
    bdofs_V = fem.locate_dofs_topological(V_space, entity_dim=tdim - 1, entities=exterior_facets)
    bc_V = fem.dirichletbc(zero_scalar, bdofs_V)
    
    if mesh.comm.rank == 0:
        print(f"Boundary DOFs: A=0 (weak BC), V={bdofs_V.size} (strong BC)")
    
    # A block has no BCs, V block has bc_V
    block_bcs = [[], [bc_V]]
    return bc_A, bc_V, block_bcs
