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
# Six copper coils (volume tags 7–12)
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
    8: {"alpha": -1.0, "beta": 0.0},
}

EXTERIOR_FACET_TAG = surface_map["Exterior"]
COIL7_LOWER_FACET_TAG = surface_map.get("Coil7Lower", None)
COIL8_LOWER_FACET_TAG = surface_map.get("Coil8Lower", None)


def load_mesh(mesh_path):
    """Load mesh + (optional) cell/facet tags from XDMF."""
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file {mesh_path} not found.")
    
    with io.XDMFFile(MPI.COMM_WORLD, str(mesh_path), "r") as xdmf:
        mesh = xdmf.read_mesh()
        # Needed for reading facet (tdim-1) MeshTags from XDMF in some DOLFINx versions
        # (avoids: "Missing IndexMap in Topology. Maybe you need to create_entities(2).")
        try:
            mesh.topology.create_entities(mesh.topology.dim - 1)
        except Exception:
            pass
        cell_tags = None
        for name in ["cell_tags", "Cell_markers", "mesh_tags", "CellTags"]:
            try:
                cell_tags = xdmf.read_meshtags(mesh, name=name)
                break
            except Exception:
                continue
        
        facet_tags = None
        for name in ["facet_tags", "Facet_markers", "facet_tags", "facets", "FacetTags"]:
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
    """Strong Dirichlet BCs: A=0 on exterior, V-driven terminals on coil ends."""
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)

    exterior_facets = (
        facet_tags.find(EXTERIOR_FACET_TAG)
        if facet_tags is not None
        else locate_entities_boundary(mesh, tdim - 1, lambda x: np.full(x.shape[1], True))
    )

    def zero_bc(space, facets):
        u0 = fem.Function(space)
        u0.x.array[:] = 0.0
        dofs = fem.locate_dofs_topological(space, tdim - 1, facets)
        return fem.dirichletbc(u0, dofs)

    # A = 0 on outer air-box boundary (as before)
    bc_A = zero_bc(A_space, exterior_facets)

    # Voltage-driven terminals on lower ends of the two active coils, if tags exist.
    bc_V_list = []
    V_terminal = 10.0

    if facet_tags is not None and COIL7_LOWER_FACET_TAG is not None:
        coil7_facets = facet_tags.find(COIL7_LOWER_FACET_TAG)
        if coil7_facets.size > 0:
            v_plus = fem.Function(V_space)
            v_plus.x.array[:] = V_terminal
            dofs_plus = fem.locate_dofs_topological(V_space, tdim - 1, coil7_facets)
            bc_V_list.append(fem.dirichletbc(v_plus, dofs_plus))

    if facet_tags is not None and COIL8_LOWER_FACET_TAG is not None:
        coil8_facets = facet_tags.find(COIL8_LOWER_FACET_TAG)
        if coil8_facets.size > 0:
            v_zero = fem.Function(V_space)
            v_zero.x.array[:] = 0.0
            dofs_zero = fem.locate_dofs_topological(V_space, tdim - 1, coil8_facets)
            bc_V_list.append(fem.dirichletbc(v_zero, dofs_zero))

    # Fallback: if no coil terminal tags were found, keep old behaviour (V=0 on exterior)
    if not bc_V_list:
        bc_V_list.append(zero_bc(V_space, exterior_facets))

    return bc_A, bc_V_list, [[bc_A], bc_V_list]
