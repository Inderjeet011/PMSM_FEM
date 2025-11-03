#!/usr/bin/env python3
"""
2D Maxwell A-V Mixed Formulation Solver
========================================
Permanent Magnet Synchronous Motor (PMSM) simulation

Formulation:
- Mixed A-V formulation (magnetic vector potential and electric scalar potential)
- Separate electrical (ω_e) and mechanical (ω_m) speeds
- Robust V constraint with rotor patch ground + regularization
- PM excitation via volume term
- σ and ω terms only in rotating conductor (rotor)

Author: Based on Abhinav's parameters and TEAM 30 techniques
"""

from dolfinx import fem, io, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from dolfinx.io import gmshio
from mpi4py import MPI
import basix.ufl
import ufl
import numpy as np
import time
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

class SimulationConfig:
    """Simulation configuration parameters"""
    # Motor geometry
    pole_pairs = 2
    n_poles = 8  # 2 * pole_pairs * 2 (N-S alternating)
    
    # Electrical parameters
    frequency = 50  # Hz (set 0 for static PM validation)
    J_peak = 7.07e6  # A/m² (peak current density) ≈ 5 A/mm² RMS
    
    # PM parameters
    B_rem = 1.4 # Tesla (remanent flux density) - INCREASED from 1.05 T
    
    # Time stepping
    dt = 0.002  # 2 ms timestep
    T_end = 0.002  # single step for static validation
    
    # Material properties
    mu0 = 4e-7 * np.pi
    
    # Derived parameters
    omega_e = 2 * np.pi * frequency
    omega_m = omega_e / pole_pairs
    M_rem = B_rem / mu0
    pole_angle_step = 2 * np.pi / n_poles
    
    @classmethod
    def print_info(cls):
        """Print configuration summary"""
        print("=" * 70)
        print(" 2D MAXWELL A-V MIXED FORMULATION (PMSM)")
        print("=" * 70)
        print(f"\n⚙️  Configuration:")
        print(f"   Pole pairs:    {cls.pole_pairs}")
        print(f"   Frequency:     {cls.frequency} Hz")
        print(f"   ω_e:           {cls.omega_e:.1f} rad/s")
        print(f"   ω_m:           {cls.omega_m:.1f} rad/s = {cls.omega_m*60/(2*np.pi):.0f} RPM")
        print(f"   J_peak:        {cls.J_peak:.3e} A/m²")
        print(f"   B_rem:         {cls.B_rem} T")
        print(f"   Timestep:      {cls.dt*1000:.2f} ms")
        print(f"   Duration:      {cls.T_end*1000:.1f} ms ({cls.T_end*cls.frequency:.0f} periods)")


class DomainTags:
    """Physical domain tags matching mesh generator"""
    OUTER_AIR = 1
    ROTOR = 2
    PM_N = 3
    PM_S = 4
    AIRGAP_INNER = 5
    AIRGAP_OUTER = 6
    STATOR = 7
    COIL_0, COIL_1, COIL_2 = 8, 9, 10
    COIL_3, COIL_4, COIL_5 = 11, 12, 13
    EXTERIOR = 100
    
    # Three-phase mapping
    COIL_AP, COIL_AM = COIL_0, COIL_3  # Phase A
    COIL_BP, COIL_BM = COIL_1, COIL_4  # Phase B
    COIL_CP, COIL_CM = COIL_2, COIL_5  # Phase C


# ============================================================================
# MAXWELL SOLVER CLASS
# ============================================================================

class MaxwellSolver2D:
    """2D Maxwell solver for PM motor using A-V mixed formulation"""
    
    def __init__(self, mesh_file="motor.msh", config=None):
        self.config = config or SimulationConfig()
        self.tags = DomainTags()
        self.mesh_file = mesh_file
        
        # Initialize storage
        self.mesh = None
        self.ct = None
        self.ft = None
        self.W = None  # Mixed function space
        self.w_prev = None
        self.w_sol = None
        
        # Material properties
        self.sigma = None
        self.nu = None
        
        # Source terms
        self.J_z = None
        self.M_x = None
        self.M_y = None
        
    def load_mesh(self):
        """Load mesh from file"""
        print("\n📖 Loading mesh...")
        self.mesh, self.ct, *rest = gmshio.read_from_msh(
            self.mesh_file, MPI.COMM_WORLD, rank=0, gdim=2
        )
        self.ft = rest[0] if rest else None
        print(f"   ✅ {self.ct.values.size} cells loaded")
        
        # Create measures
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.ct)
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.ft) if self.ft else ufl.ds
        
    def setup_materials(self):
        """Setup material properties (conductivity and reluctivity)"""
        print("\n🔧 Setting up materials...")
        
        DG0 = fem.functionspace(self.mesh, ("DG", 0))
        mu0 = self.config.mu0
        
        # Conductivity (σ) - only rotor conducts
        sigma_vals = {
            self.tags.OUTER_AIR: 0.0,
            self.tags.AIRGAP_INNER: 0.0,
            self.tags.AIRGAP_OUTER: 0.0,
            self.tags.ROTOR: 2e6,  # Rotor conductivity
            self.tags.PM_N: 0.0,
            self.tags.PM_S: 0.0,
            self.tags.STATOR: 0.0,
            self.tags.COIL_AP: 0.0,
            self.tags.COIL_AM: 0.0,
            self.tags.COIL_BP: 0.0,
            self.tags.COIL_BM: 0.0,
            self.tags.COIL_CP: 0.0,
            self.tags.COIL_CM: 0.0,
        }
        
        # Reluctivity (ν = 1/μ)
        nu_vals = {
            self.tags.OUTER_AIR: 1/mu0,
            self.tags.AIRGAP_INNER: 1/mu0,
            self.tags.AIRGAP_OUTER: 1/mu0,
            self.tags.ROTOR: 1/(mu0*1000),  # High permeability - INCREASED from 100
            self.tags.PM_N: 1/mu0,
            self.tags.PM_S: 1/mu0,
            self.tags.STATOR: 1/(mu0*1000),  # High permeability - INCREASED from 100
            self.tags.COIL_AP: 1/(mu0*0.999991),
            self.tags.COIL_AM: 1/(mu0*0.999991),
            self.tags.COIL_BP: 1/(mu0*0.999991),
            self.tags.COIL_BM: 1/(mu0*0.999991),
            self.tags.COIL_CP: 1/(mu0*0.999991),
            self.tags.COIL_CM: 1/(mu0*0.999991),
        }
        
        # Create material functions
        self.sigma = fem.Function(DG0)
        self.nu = fem.Function(DG0)
        
        for tag in sigma_vals:
            cells = self.ct.find(tag)
            if len(cells):
                self.sigma.x.array[cells] = sigma_vals[tag]
                self.nu.x.array[cells] = nu_vals[tag]
        
        print("   ✅ Materials assigned (σ, ν)")
        
    def setup_function_spaces(self):
        """Setup mixed function space (Az, V)"""
        print("\n🎯 Setting up function spaces...")
        
        # Mixed space: P1 for Az (magnetic vector potential) and V (electric potential)
        P1A = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1)
        P1V = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1)
        self.W = fem.functionspace(
            self.mesh, basix.ufl.mixed_element([P1A, P1V])
        )
        
        print(f"   ✅ Mixed DOFs: {self.W.dofmap.index_map.size_global * self.W.dofmap.index_map_bs}")
        
        # Initialize solution functions
        self.w_prev = fem.Function(self.W)
        self.w_sol = fem.Function(self.W)
        
    def setup_boundary_conditions(self):
        """Setup boundary conditions"""
        print("\n🔒 Setting up boundary conditions...")
        
        tdim = self.mesh.topology.dim
        
        # BC 1: Az = 0 on exterior boundary
        W0, _ = self.W.sub(0).collapse()
        
        if self.ft is not None:
            exterior_facets = self.ft.find(self.tags.EXTERIOR)
            if len(exterior_facets) > 0:
                bnd = exterior_facets
                print(f"   ✅ Az=0 on EXTERIOR: {len(bnd)} facets")
            else:
                bnd = locate_entities_boundary(
                    self.mesh, tdim-1, lambda X: np.full(X.shape[1], True)
                )
                print(f"   ⚠️  EXTERIOR tag not found, using all boundaries")
        else:
            bnd = locate_entities_boundary(
                self.mesh, tdim-1, lambda X: np.full(X.shape[1], True)
            )
            print(f"   Using all exterior boundaries: {len(bnd)} facets")
        
        bdofs_Az = fem.locate_dofs_topological((self.W.sub(0), W0), tdim-1, bnd)
        zeroAz = fem.Function(W0)
        zeroAz.x.array[:] = 0.0
        self.bcAz = fem.dirichletbc(zeroAz, bdofs_Az, self.W.sub(0))
        
        # BC 2: V = 0 on rotor patch (robust ground)
        W1, _ = self.W.sub(1).collapse()
        self.mesh.topology.create_connectivity(tdim, 0)
        self.mesh.topology.create_connectivity(0, tdim)
        c2v = self.mesh.topology.connectivity(tdim, 0)
        
        rotor_cells = self.ct.find(self.tags.ROTOR)
        assert rotor_cells.size > 0, "No rotor cells found for V grounding"
        
        # Use first 30 rotor cells as patch ground
        patch_cells = np.array(rotor_cells[:min(30, rotor_cells.size)], dtype=np.int32)
        patch_verts = np.unique(
            np.hstack([c2v.links(int(c)) for c in patch_cells])
        ).astype(np.int32)
        
        vdofs = fem.locate_dofs_topological((self.W.sub(1), W1), 0, patch_verts)
        V0 = fem.Function(W1)
        V0.x.array[:] = 0.0
        self.bcV = fem.dirichletbc(V0, vdofs, self.W.sub(1))
        
        print(f"   ✅ V=0 on rotor patch: {len(vdofs)} DOFs")
        
        self.bcs = [self.bcAz, self.bcV]
        
    def initialize_sources(self):
        """Initialize source terms (currents and magnetization)"""
        print("\n⚡ Initializing sources...")
        
        DG0 = fem.functionspace(self.mesh, ("DG", 0))
        
        # Current density
        self.J_z = fem.Function(DG0)
        print(f"   ✅ Current density: J_peak = {self.config.J_peak:.3e} A/m²")
        
        # Magnetization
        self.M_x = fem.Function(DG0)
        self.M_y = fem.Function(DG0)
        
        # Initialize PM magnetization
        for pm_tag, sign in [(self.tags.PM_N, +1), (self.tags.PM_S, -1)]:
            cells = self.ct.find(pm_tag)
            if cells.size == 0:
                continue
            
            for c in cells:
                # Get cell centroid
                cell_geom_dofs = self.mesh.geometry.dofmap[c]
                cx = np.mean(self.mesh.geometry.x[cell_geom_dofs, 0])
                cy = np.mean(self.mesh.geometry.x[cell_geom_dofs, 1])
                
                # Find angular position
                theta = np.arctan2(cy, cx)
                if theta < 0:
                    theta += 2 * np.pi
                
                # Find which pole and center on it
                pole_idx = int(np.round(theta / self.config.pole_angle_step))
                theta_pole_center = (pole_idx + 0.5) * self.config.pole_angle_step
                
                # Magnetization points radially
                self.M_x.x.array[c] = sign * self.config.M_rem * np.cos(theta_pole_center)
                self.M_y.x.array[c] = sign * self.config.M_rem * np.sin(theta_pole_center)
        
        print(f"   ✅ Magnetization: B_rem = {self.config.B_rem} T, {self.config.n_poles} poles")
        
    def create_variational_form(self):
        """Create variational formulation"""
        print("\n📐 Creating variational form...")
        
        # Domain helpers
        def dxc(tags):
            """Create measure over multiple subdomains"""
            if not tags:
                return self.dx(999)  # Non-existent tag
            m = self.dx(tags[0])
            for t in tags[1:]:
                m += self.dx(t)
            return m
        
        # Conducting region (rotor only)
        Omega_conducting = [self.tags.ROTOR]
        
        # Coil regions
        Omega_coils = [
            self.tags.COIL_AP, self.tags.COIL_AM,
            self.tags.COIL_BP, self.tags.COIL_BM,
            self.tags.COIL_CP, self.tags.COIL_CM
        ]
        
        # PM regions
        Omega_pm = [self.tags.PM_N, self.tags.PM_S]
        
        # Trial and test functions
        (Az, V) = ufl.TrialFunctions(self.W)
        (v, q) = ufl.TestFunctions(self.W)
        
        # Previous solution
        Az_prev, V_prev = ufl.split(self.w_prev)
        
        # Spatial coordinates and rotation velocity
        xcoord = ufl.SpatialCoordinate(self.mesh)
        x, y = xcoord[0], xcoord[1]
        
        omega = fem.Constant(self.mesh, self.config.omega_m)
        dt = fem.Constant(self.mesh, self.config.dt)
        mu0 = self.config.mu0
        
        # Rotation velocity field
        u_rot_x = -omega * y
        u_rot_y = omega * x
        
        # Regularization for V
        epsV = fem.Constant(self.mesh, 1e-12)
        
        # === Bilinear form a(Az, V; v, q) ===
        
        # Magnetic diffusion term
        a = self.nu * ufl.inner(ufl.grad(Az), ufl.grad(v)) * self.dx
        
        # Motional term in conductor (convection)
        a += self.sigma * (u_rot_x*ufl.grad(Az)[0] + u_rot_y*ufl.grad(Az)[1]) * v * dxc(Omega_conducting)
        
        # Time-discrete coupling (A-block)
        a += (self.sigma/dt) * Az * v * dxc(Omega_conducting)
        
        # V-equation in conductor
        a += self.sigma * ufl.inner(ufl.grad(V), ufl.grad(q)) * dxc(Omega_conducting)
        a += (self.sigma/dt) * Az * q * dxc(Omega_conducting)
        
        # V regularization
        a += epsV * V * q * dxc(Omega_conducting)
        
        # Boundary term (optional)
        n = ufl.FacetNormal(self.mesh)
        a += (1e-16) * v * (n[0]*Az.dx(0) - n[1]*Az.dx(1)) * self.ds
        
        # === Linear form L(v, q) ===
        
        # Time-discrete term from previous solution
        L = (self.sigma/dt) * Az_prev * v * dxc(Omega_conducting)
        
        # Coil source terms
        for coil in Omega_coils:
            L += self.J_z * v * self.dx(coil)
        
        # PM source term: -∫ M · curl(v) dΩ over PM regions
        curl_v = ufl.as_vector((v.dx(1), -v.dx(0)))
        M_vec = ufl.as_vector((self.M_x, self.M_y))
        L += -ufl.inner(M_vec, curl_v) * dxc(Omega_pm)
        
        # V-equation RHS (motional term)
        vXB_prev = omega * (y*ufl.grad(Az_prev)[0] - x*ufl.grad(Az_prev)[1])
        L += (self.sigma/dt) * Az_prev * q * dxc(Omega_conducting)
        L += -self.sigma * vXB_prev * q * dxc(Omega_conducting)
        
        print("   ✅ Variational form created")
        
        self.a = a
        self.L = L
        
    def update_currents(self, t):
        """Update three-phase currents"""
        omega_e = self.config.omega_e
        J_peak = self.config.J_peak
        
        # Three-phase currents
        IA = J_peak * np.sin(omega_e * t)
        IB = J_peak * np.sin(omega_e * t - 2*np.pi/3)
        IC = J_peak * np.sin(omega_e * t + 2*np.pi/3)
        
        # Assign to coils
        for c in self.ct.find(self.tags.COIL_AP): self.J_z.x.array[c] = IA
        for c in self.ct.find(self.tags.COIL_AM): self.J_z.x.array[c] = -IA
        for c in self.ct.find(self.tags.COIL_BP): self.J_z.x.array[c] = IB
        for c in self.ct.find(self.tags.COIL_BM): self.J_z.x.array[c] = -IB
        for c in self.ct.find(self.tags.COIL_CP): self.J_z.x.array[c] = IC
        for c in self.ct.find(self.tags.COIL_CM): self.J_z.x.array[c] = -IC
        
    def rotate_magnetization(self, t):
        """Rotate PM magnetization with rotor"""
        theta_rot = self.config.omega_m * t
        
        for pm_tag, sign in [(self.tags.PM_N, +1), (self.tags.PM_S, -1)]:
            cells = self.ct.find(pm_tag)
            for c in cells:
                # Get cell centroid
                cell_geom_dofs = self.mesh.geometry.dofmap[c]
                cx = np.mean(self.mesh.geometry.x[cell_geom_dofs, 0])
                cy = np.mean(self.mesh.geometry.x[cell_geom_dofs, 1])
                
                # Find angular position
                theta = np.arctan2(cy, cx)
                if theta < 0:
                    theta += 2 * np.pi
                
                # Find pole center
                pole_idx = int(np.round(theta / self.config.pole_angle_step))
                theta_pole_center = (pole_idx + 0.5) * self.config.pole_angle_step
                
                # Rotate magnetization
                theta_now = theta_pole_center + theta_rot
                self.M_x.x.array[c] = sign * self.config.M_rem * np.cos(theta_now)
                self.M_y.x.array[c] = sign * self.config.M_rem * np.sin(theta_now)
    
    def solve(self, output_file="../results/results_2d_mixed.xdmf"):
        """Main time-stepping solver"""
        print("\n⏱️  Starting time-stepping simulation...")
        
        num_steps = int(self.config.T_end / self.config.dt)
        print(f"   Steps: {num_steps}")
        print(f"   Duration: {self.config.T_end*1000:.1f} ms")
        print()
        
        # Output file - create results directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        out = io.XDMFFile(self.mesh.comm, output_file, "w")
        out.write_mesh(self.mesh)
        # Write cell tags if available (needed for post-processing like torque)
        if self.ct is not None:
            try:
                out.write_meshtags(self.ct)
            except Exception:
                pass
        
        # PETSc options for solver
        petsc_options = {
            "ksp_type": "gmres",
            "ksp_rtol": 1e-8,
            "ksp_error_if_not_converged": True,
            "pc_type": "lu",
            "pc_factor_shift_type": "NONZERO",
        }
        
        # Time loop
        start_time = time.time()
        t = 0.0
        
        for step in range(1, num_steps + 1):
            t += self.config.dt
            
            # Update sources
            self.update_currents(t)
            self.rotate_magnetization(t)
            
            # Solve linear system
            prob = LinearProblem(
                self.a, self.L, bcs=self.bcs,
                petsc_options=petsc_options,
                petsc_options_prefix="mixed"
            )
            self.w_sol = prob.solve()
            
            # Extract components
            Az_sol, V_sol = self.w_sol.split()
            
            # Update previous solution
            self.w_prev.x.array[:] = self.w_sol.x.array[:]
            
            # Save output periodically
            if step % max(1, num_steps // 8) == 0:
                out.write_function(Az_sol, t)
            
            # Progress
            norm_Az = np.linalg.norm(Az_sol.x.array)
            print(f"   Step {step:3d}/{num_steps}  t={t*1e3:5.2f} ms  ||Az||={norm_Az:.2e}")
            # Report final-step average magnetic flux density magnitude (B_rms)
            if step == num_steps:
                Bx = ufl.grad(Az_sol)[1]
                By = -ufl.grad(Az_sol)[0]
                b2 = Bx*Bx + By*By
                # Whole-domain RMS (diagnostic)
                area_all = fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=self.mesh)))
                val_all = fem.assemble_scalar(fem.form(b2 * ufl.dx(domain=self.mesh)))
                Brms_all = float(np.sqrt(val_all / max(area_all, 1e-18)))
                # Air-gap-only RMS
                import math as _math
                def _cell_r(c: int) -> float:
                    g = self.mesh.geometry.dofmap[c]
                    cx = float(np.mean(self.mesh.geometry.x[g, 0]))
                    cy = float(np.mean(self.mesh.geometry.x[g, 1]))
                    return _math.hypot(cx, cy)
                pm_cells = np.concatenate([self.ct.find(self.tags.PM_N), self.ct.find(self.tags.PM_S)])
                stator_cells = self.ct.find(self.tags.STATOR)
                if pm_cells.size and stator_cells.size:
                    rin = max(_cell_r(int(c)) for c in pm_cells)
                    rout = min(_cell_r(int(c)) for c in stator_cells)
                else:
                    rin, rout = 0.0, 1.0
                xcoord = ufl.SpatialCoordinate(self.mesh)
                r = ufl.sqrt(xcoord[0]**2 + xcoord[1]**2)
                mask = ufl.conditional(ufl.lt(r, rout), ufl.conditional(ufl.gt(r, rin), 1.0, 0.0), 0.0)
                area_gap = fem.assemble_scalar(fem.form(mask * ufl.dx(domain=self.mesh)))
                val_gap = fem.assemble_scalar(fem.form(b2 * mask * ufl.dx(domain=self.mesh)))
                Brms_gap = float(np.sqrt(val_gap / max(area_gap, 1e-18)))
                print(f"   Final B_rms (domain) ≈ {Brms_all:.3e} T, B_rms (air-gap) ≈ {Brms_gap:.3e} T")
        
        out.close()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Simulation complete in {elapsed:.1f}s")
        print(f"   Output: {output_file}")
        
    def run(self):
        """Execute full simulation workflow"""
        self.config.print_info()
        
        self.load_mesh()
        self.setup_materials()
        self.setup_function_spaces()
        self.setup_boundary_conditions()
        self.initialize_sources()
        self.create_variational_form()
        self.solve()
        
        print("\n" + "=" * 70)
        print(" ✅ MAXWELL SOLVER COMPLETE")
        print("=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    solver = MaxwellSolver2D(mesh_file="../motor.msh")
    solver.run()

