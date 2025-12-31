"""Setup functions for 3D solver."""
import numpy as np
from dolfinx import fem, io
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from mesh_3D import model_parameters, surface_map


# Domain tags
AIR = (1,)
AIR_GAP = (2, 3)
ALUMINIUM = (4,)
ROTOR = (5,)
STATOR = (6,)
COILS = (7, 8, 9, 10, 11, 12)
MAGNETS = (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)

def conducting():
    # Conducting regions for σ-terms and the V-equation (include coils).
    return ROTOR + ALUMINIUM + MAGNETS + COILS

# ---------------------------------------------------------------------------
# r = rotor, s = stator, pm = permanent magnets, c = coils.
# Shaft/center-rod (ALUMINIUM) is treated as part of the rotor.
# ---------------------------------------------------------------------------

def omega_r():
    """Ω_r: rotor assembly (rotor + shaft/rod)."""
    return ROTOR + ALUMINIUM


def omega_s():
    """Ω_s: stator."""
    return STATOR


def omega_pm():
    """Ω_pm: permanent magnets."""
    return MAGNETS


def omega_c():
    """Ω_c: coils."""
    return COILS


def omega_rs():
    """Ω_{r,s} = Ω_r ∪ Ω_s."""
    return omega_r() + omega_s()


def omega_rpm():
    """Ω_{r,pm} = Ω_r ∪ Ω_pm."""
    return omega_r() + omega_pm()

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


def load_mesh(mesh_path):
    """Load mesh + (optional) cell/facet tags from XDMF."""
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file {mesh_path} not found.")
    
    with io.XDMFFile(MPI.COMM_WORLD, str(mesh_path), "r") as xdmf:
        mesh = xdmf.read_mesh()
        cell_tags = None
        for name in ["Cell_markers", "mesh_tags"]:
            try:
                cell_tags = xdmf.read_meshtags(mesh, name=name)
                break
            except Exception:
                continue
        
        facet_tags = None
        for name in ["Facet_markers", "facet_tags"]:
            try:
                facet_tags = xdmf.read_meshtags(mesh, name=name)
                break
            except Exception:
                continue
    
    if mesh.comm.rank == 0:
        print(f"Mesh loaded: {mesh.topology.index_map(mesh.topology.dim).size_global} cells")
    
    return mesh, cell_tags, facet_tags


def setup_materials(mesh, cell_tags, config):
    """Material parameters as DG0 Functions (sigma, nu, density)."""
    DG0 = fem.functionspace(mesh, ("DG", 0))
    sigma = fem.Function(DG0, name="sigma")       #array of sigma values for each cell
    nu = fem.Function(DG0, name="nu")             #array of reluctivity values for each cell
    density = fem.Function(DG0, name="density")   #array of density values for each cell
    
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
        **{m: "Cu" for m in COILS},
        **{m: "PM" for m in MAGNETS},
    }
    
    mu0 = config.mu0
    for marker, mat_name in marker_to_material.items():
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        sigma.x.array[cells] = sigma_dict[mat_name]
        density.x.array[cells] = densities.get(mat_name, 0.0)
        nu.x.array[cells] = 1.0 / (mu0 * mu_r[mat_name])
    
    if mesh.comm.rank == 0:
        print("Materials assigned")
    
    return sigma, nu, density


def setup_boundary_conditions(mesh, facet_tags, A_space, V_space):
    """Strong Dirichlet BCs (A=0, V=0) on the exterior boundary."""
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)

    exterior_facets = (
        facet_tags.find(EXTERIOR_FACET_TAG)
        if facet_tags is not None
        else locate_entities_boundary(mesh, tdim - 1, lambda x: np.full(x.shape[1], True))
    )

    def zero_bc(space):
        u0 = fem.Function(space)
        u0.x.array[:] = 0.0
        dofs = fem.locate_dofs_topological(space, tdim - 1, exterior_facets)
        return fem.dirichletbc(u0, dofs)

    bc_A = zero_bc(A_space)
    bc_V = zero_bc(V_space)
    return bc_A, bc_V, [[bc_A], [bc_V]]
