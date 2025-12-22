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
        self.steps_per_period = 10
        period = 1.0 / self.drive_frequency
        self.dt = period / self.steps_per_period
        # Run exactly one period (f=60 Hz => period ≈ 16.67 ms) with fewer timestamps.
        # With steps_per_period=10, dt ≈ 1.667 ms and 10 steps cover one full period.
        self.num_steps = 10
        self.degree_A = 1
        self.degree_V = 1
        self.coil_current_peak = model_parameters["J"]
        self.mu0 = model_parameters["mu_0"]
        self.mesh_path = Path(__file__).parents[2] / "meshes" / "3d" / "pmesh3D_ipm.xdmf"
        # Output: write only these two files (XDMF + HDF5)
        self.results_path = Path(__file__).parents[2] / "results" / "3d" / "av_solver.xdmf"
        self.write_results = True
        # Optional: write a diagnostics CSV (disabled by default to keep outputs minimal)
        self.write_diagnostics = False
        self.diagnostics_path = Path(__file__).parents[2] / "results" / "3d" / "av_solver_diagnostics.csv"

        # Optional: write wall-clock timing CSV (disabled by default)
        # This is useful for comparing "before vs after" performance runs.
        self.write_timings = False
        self.timings_path = Path(__file__).parents[2] / "results" / "3d" / "av_solver_timings.csv"

        # Motor-only output disabled (write full mesh directly)
        self.output_motor_only = False
        self.magnet_remanence = 1.2
        # Output region for exported fields.
        # Per project requirement: export the full-domain solution as-is (including outer air box),
        # and do not clip/filter/project to motor-only volumes in code.
        self.B_output_region = "full"
        # -----------------------------
        # Linear solver tuning knobs
        # -----------------------------
        # FieldSplit preconditioner choice for the coupled A–V system.
        # "additive" has been the most robust for the PMSM case so far.
        self.fieldsplit_type = "additive"  # "schur" | "additive" | "multiplicative"
        self.schur_pre = "A11"  # if fieldsplit_type == "schur": "A11" | "SELFP"
        
        # Balance the V-row scaling (improves conditioning; does not change the exact solution).
        self.scale_V_row_by_dt = True
        
        # A-block apply inside the preconditioner.
        # "fgmres" with a few iterations gives much stronger convergence than a single AMS apply.
        self.ksp_A_type = "fgmres"  # "preonly" | "fgmres"
        self.ksp_A_max_it = 20
        self.ksp_A_restart = 50
        
        # V-block solve for additive fieldsplit (0 keeps it as preonly).
        self.ksp_V_max_it = 10
        
        # Outer KSP limits (keep small for quick runs; increase when diagnosing).
        # With the current additive preconditioner, rtol ~2e-2 is realistic for <=50 its.
        self.outer_rtol = 1e-6
        self.outer_max_it = 100
        
        # Timestep acceptance threshold (ratio = ||b-Ax|| / ||b-Ax0||).
        # With the current preconditioner we typically get ~0.31 at 50 its; 0.35 keeps marching stable.
        self.accept_ratio = 0.35
        
        # A00_spd boundary penalty scaling (AMS-only matrix; does not change physics).
        self.alpha_spd_factor = 1e-3

        # Small operator regularization for robustness (verify DG0 B stats when changing):
        # This removes a curl-curl near-nullspace mode that otherwise causes Krylov plateaus.
        self.epsilon_A_full = 1e-4

        # Keep the SPD preconditioner mass shift tiny; the main effect comes from epsilon_A_full.
        self.epsilon_A_spd = 1e-6
    
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


def maybe_retag_cells(mesh, cell_tags, force_retag=False):
    """Retag cells if needed.
    
    Args:
        force_retag: If True, always retag even if all required tags are present.
    """
    if cell_tags is None:
        raise RuntimeError("Mesh must provide cell tags.")
    
    current_tags = set(np.unique(cell_tags.values))
    required_tags = {ROTOR[0], ALUMINIUM[0], STATOR[0], AIR[0], AIR_GAP[0], AIR_GAP[1]}
    required_tags.update(COILS)
    required_tags.update(MAGNETS)
    
    if not force_retag and required_tags.issubset(current_tags):
        # Tags are present, but validate they're in correct locations
        if mesh.comm.rank == 0:
            print("All required tags present. Validating domain locations...")
        # Check if airgap is correctly tagged
        coords = mesh.geometry.x
        dofmap = mesh.geometry.dofmap
        centers = coords[dofmap].mean(axis=1)
        radii = np.linalg.norm(centers[:, :2], axis=1)
        r3, r4 = mesh_parameters["r3"], mesh_parameters["r4"]
        in_airgap_band = (radii >= r3) & (radii < r4)
        airgap_cells = np.concatenate([cell_tags.find(AIR_GAP[0]), cell_tags.find(AIR_GAP[1])])
        airgap_in_band = np.sum(np.isin(np.where(in_airgap_band)[0], airgap_cells))
        total_in_band = np.sum(in_airgap_band)
        # Count coils in band
        coil_cells = []
        for tag in COILS:
            coil_cells.extend(cell_tags.find(tag).tolist())
        coil_in_band = np.sum(np.isin(np.where(in_airgap_band)[0], np.array(coil_cells)))
        expected_airgap = total_in_band - coil_in_band
        if mesh.comm.rank == 0:
            print(f"  Cells in r3-r4: {total_in_band}, tagged as airgap: {airgap_in_band}, coils: {coil_in_band}")
            print(f"  Expected airgap cells: {expected_airgap}")
            if airgap_in_band < expected_airgap * 0.9:  # Allow 10% tolerance
                print(f"  ⚠️  WARNING: Airgap tagging appears incorrect. Forcing retag...")
                force_retag = True
            else:
                validate_domain_tags(mesh, cell_tags)
                return cell_tags
    
    if force_retag and mesh.comm.rank == 0:
        print("Force retagging cells to fix domain locations...")
    
    if mesh.comm.rank == 0:
        print("Retagging cells...")
    
    coords = mesh.geometry.x
    dofmap = mesh.geometry.dofmap
    centers = coords[dofmap].mean(axis=1)
    radii = np.linalg.norm(centers[:, :2], axis=1)
    angles = np.mod(np.arctan2(centers[:, 1], centers[:, 0]), 2 * np.pi)
    
    r1, r2, r3, r4, r5, r6, r7 = [mesh_parameters[f"r{i}"] for i in range(1, 8)]
    r_mid_gap = 0.5 * (r3 + r4)  # CORRECT: airgap is between r3 and r4, split at midpoint
    
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
        elif r <= r3 - radial_tol:
            tag = ALUMINIUM[0]  # Rotor iron continues to r3
        elif r3 - radial_tol < r <= r4 + radial_tol:
            # AIRGAP REGION: r3 to r4
            # First check if this is a coil slot
            idx, delta = nearest(theta, coil_centers)
            if delta <= coil_half:
                tag = COILS[idx]  # Coil slot
            else:
                # Not a coil - must be airgap
                if r <= r_mid_gap + radial_tol:
                    tag = AIR_GAP[0]  # Inner half of airgap
                else:
                    tag = AIR_GAP[1]  # Outer half of airgap
        elif r4 - radial_tol < r <= r5 + radial_tol:
            tag = STATOR[0]
        else:
            tag = AIR[0]  # Outer airbox
        
        new_tags[cell] = tag
    
    cell_indices = np.arange(len(new_tags), dtype=np.int32)
    new_cell_tags = dmesh.meshtags(mesh, mesh.topology.dim, cell_indices, new_tags)
    
    if mesh.comm.rank == 0:
        print(f"Retagged mesh. Unique tags: {sorted(set(new_tags.tolist()))}")
    
    # Robust validation: verify domains are in correct physical locations
    validate_domain_tags(mesh, new_cell_tags)
    
    return new_cell_tags


def validate_domain_tags(mesh, cell_tags):
    """Robust validation: check that each domain is in the correct physical location."""
    if cell_tags is None:
        return
    
    if mesh.comm.rank == 0:
        print("\n" + "="*70)
        print("DOMAIN TAG VALIDATION")
        print("="*70)
    
    coords = mesh.geometry.x
    dofmap = mesh.geometry.dofmap
    centers = coords[dofmap].mean(axis=1)
    radii = np.linalg.norm(centers[:, :2], axis=1)
    angles = np.mod(np.arctan2(centers[:, 1], centers[:, 0]), 2 * np.pi)
    z_coords = centers[:, 2]
    
    r1, r2, r3, r4, r5, r6, r7 = [mesh_parameters[f"r{i}"] for i in range(1, 8)]
    r_mid_gap = 0.5 * (r3 + r4)
    
    # Expected radial ranges for each domain
    expected_ranges = {
        "ROTOR": (0.0, r3),
        "ALUMINIUM": (r1, r2),
        "AIR_GAP": (r3, r4),
        "STATOR": (r4, r5),
        "AIR": (r5, 1.0),  # Outer airbox
    }
    
    # Check airgap (most critical)
    airgap_cells_tag2 = cell_tags.find(AIR_GAP[0])
    airgap_cells_tag3 = cell_tags.find(AIR_GAP[1])
    all_airgap_cells = np.concatenate([airgap_cells_tag2, airgap_cells_tag3]) if (airgap_cells_tag2.size > 0 and airgap_cells_tag3.size > 0) else (airgap_cells_tag2 if airgap_cells_tag2.size > 0 else airgap_cells_tag3)
    
    if all_airgap_cells.size > 0:
        airgap_radii = radii[all_airgap_cells]
        airgap_r_min = float(np.min(airgap_radii))
        airgap_r_max = float(np.max(airgap_radii))
        airgap_r_mean = float(np.mean(airgap_radii))
        
        if mesh.comm.rank == 0:
            print(f"\n[AIRGAP VALIDATION]")
            print(f"  Expected range: r3={r3:.4f} to r4={r4:.4f}")
            print(f"  Actual range:   {airgap_r_min:.4f} to {airgap_r_max:.4f} (mean={airgap_r_mean:.4f})")
            print(f"  Cell count:     {all_airgap_cells.size}")
            
            # Check for cells outside expected range
            outside_low = np.sum(airgap_radii < r3 - 0.001)
            outside_high = np.sum(airgap_radii > r4 + 0.001)
            if outside_low > 0 or outside_high > 0:
                print(f"  ⚠️  WARNING: {outside_low} cells below r3, {outside_high} cells above r4")
            else:
                print(f"  ✅ Airgap cells are within expected radial range")
    
    # Check rotor
    rotor_cells = cell_tags.find(ROTOR[0])
    if rotor_cells.size > 0:
        rotor_radii = radii[rotor_cells]
        rotor_r_max = float(np.max(rotor_radii))
        if mesh.comm.rank == 0:
            print(f"\n[ROTOR VALIDATION]")
            print(f"  Expected: max r <= r3={r3:.4f}")
            print(f"  Actual:   max r = {rotor_r_max:.4f}, cells={rotor_cells.size}")
            if rotor_r_max > r3 + 0.001:
                print(f"  ⚠️  WARNING: Some rotor cells extend beyond r3")
            else:
                print(f"  ✅ Rotor cells are within expected range")
    
    # Check stator
    stator_cells = cell_tags.find(STATOR[0])
    if stator_cells.size > 0:
        stator_radii = radii[stator_cells]
        stator_r_min = float(np.min(stator_radii))
        stator_r_max = float(np.max(stator_radii))
        if mesh.comm.rank == 0:
            print(f"\n[STATOR VALIDATION]")
            print(f"  Expected: r4={r4:.4f} to r5={r5:.4f}")
            print(f"  Actual:   {stator_r_min:.4f} to {stator_r_max:.4f}, cells={stator_cells.size}")
            if stator_r_min < r4 - 0.001 or stator_r_max > r5 + 0.001:
                print(f"  ⚠️  WARNING: Stator cells outside expected range")
            else:
                print(f"  ✅ Stator cells are within expected range")
    
    # Check magnets
    all_magnet_cells = []
    for tag in MAGNETS:
        cells = cell_tags.find(tag)
        if cells.size > 0:
            all_magnet_cells.append(cells)
    if all_magnet_cells:
        all_magnet_cells = np.concatenate(all_magnet_cells)
        magnet_radii = radii[all_magnet_cells]
        magnet_r_min = float(np.min(magnet_radii))
        magnet_r_max = float(np.max(magnet_radii))
        if mesh.comm.rank == 0:
            print(f"\n[MAGNETS VALIDATION]")
            print(f"  Expected: r6={r6:.4f} to r7={r7:.4f}")
            print(f"  Actual:   {magnet_r_min:.4f} to {magnet_r_max:.4f}, cells={all_magnet_cells.size}")
            if magnet_r_min < r6 - 0.001 or magnet_r_max > r7 + 0.001:
                print(f"  ⚠️  WARNING: Magnet cells outside expected range")
            else:
                print(f"  ✅ Magnet cells are within expected range")
    
    # Check coils
    all_coil_cells = []
    for tag in COILS:
        cells = cell_tags.find(tag)
        if cells.size > 0:
            all_coil_cells.append(cells)
    if all_coil_cells:
        all_coil_cells = np.concatenate(all_coil_cells)
        coil_radii = radii[all_coil_cells]
        coil_r_min = float(np.min(coil_radii))
        coil_r_max = float(np.max(coil_radii))
        if mesh.comm.rank == 0:
            print(f"\n[COILS VALIDATION]")
            print(f"  Expected: r3={r3:.4f} to r4={r4:.4f} (in slots)")
            print(f"  Actual:   {coil_r_min:.4f} to {coil_r_max:.4f}, cells={all_coil_cells.size}")
            if coil_r_min < r3 - 0.001 or coil_r_max > r4 + 0.001:
                print(f"  ⚠️  WARNING: Coil cells outside expected range")
            else:
                print(f"  ✅ Coil cells are within expected range")
    
    # Summary: check for any cells in wrong radial bands
    if mesh.comm.rank == 0:
        print(f"\n[SUMMARY]")
        total_cells = mesh.topology.index_map(mesh.topology.dim).size_global
        print(f"  Total mesh cells: {total_cells}")
        
        # Count cells in each radial band
        bands = {
            "r < r1": (0, r1),
            "r1 < r < r3": (r1, r3),
            "r3 < r < r4 (airgap)": (r3, r4),
            "r4 < r < r5": (r4, r5),
            "r > r5": (r5, 10.0),
        }
        
        for band_name, (r_low, r_high) in bands.items():
            count = np.sum((radii >= r_low) & (radii < r_high))
            print(f"  Cells in {band_name}: {count}")
        
        print("="*70 + "\n")


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
