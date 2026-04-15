#!/usr/bin/env python3
"""2D transient A–V (Az, V) Maxwell solver for PMSM: Gmsh mesh, σ/ν regions, PM + 3-phase coils."""

from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
try:
    from dolfinx.io import gmshio
except ImportError:
    from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
import basix.ufl
import ufl
import numpy as np
import time
import math
from pathlib import Path


# --- Configuration (time grid, excitation, output policy) ---

class SimulationConfig:
    def __init__(self, 
                 pole_pairs=5,
                 frequency=50,
                 J_peak=7.07e6,
                 B_rem=1.4,
                 T_end=0.010,
                 num_steps=100,
                 dt=None,
                 polynomial_degree=1,
                 omega_m=None,
                 output_num_timestamps=100,
                 write_every_timestep=True):
        self.pole_pairs = pole_pairs
        self.n_poles = 2 * pole_pairs
        self.frequency = frequency
        self.J_peak = J_peak
        self.B_rem = B_rem
        self.T_end = float(T_end)
        self.num_steps = None
        self.dt = None
        self._set_time_grid(num_steps=num_steps, dt=dt)
        self.output_num_timestamps = int(output_num_timestamps)
        self.write_every_timestep = bool(write_every_timestep)
        self.mu0 = 4e-7 * np.pi
        self.polynomial_degree = polynomial_degree
        self._omega_e = None
        self._omega_m = None
        self._M_rem = None
        self._pole_angle_step = None
        self._omega_m_override = omega_m
        self._update_derived()

    def _set_time_grid(self, num_steps=None, dt=None):
        if self.T_end <= 0.0:
            raise ValueError("T_end must be positive.")

        if num_steps is not None:
            n = int(num_steps)
            if n <= 0:
                raise ValueError("num_steps must be positive.")
            self.num_steps = n
            self.dt = self.T_end / self.num_steps
            return

        if dt is not None:
            dt = float(dt)
            if dt <= 0.0:
                raise ValueError("dt must be positive.")
            self.num_steps = max(1, int(round(self.T_end / dt)))
            self.dt = self.T_end / self.num_steps
            return

        raise ValueError("Provide either num_steps or dt.")
    
    def _update_derived(self):
        self._omega_e = 2 * np.pi * self.frequency
        if self._omega_m_override is not None:
            self._omega_m = self._omega_m_override
        else:
            self._omega_m = self._omega_e / self.pole_pairs
        self._M_rem = self.B_rem / self.mu0
        self._pole_angle_step = 2 * np.pi / self.n_poles
    
    @property
    def omega_e(self):
        return self._omega_e
    
    @property
    def omega_m(self):
        return self._omega_m
    
    @property
    def M_rem(self):
        return self._M_rem
    
    @property
    def pole_angle_step(self):
        return self._pole_angle_step
    
    def print_info(self):
        """Print configuration summary"""
        print("=" * 70)
        print(" 2D MAXWELL A-V MIXED FORMULATION (PMSM)")
        print("=" * 70)
        print(f"\n⚙️  Configuration:")
        print(f"   Pole pairs:    {self.pole_pairs}")
        print(f"   Frequency:     {self.frequency} Hz")
        print(f"   ω_e:           {self.omega_e:.1f} rad/s")
        print(f"   ω_m:           {self.omega_m:.1f} rad/s = {self.omega_m*60/(2*np.pi):.0f} RPM")
        print(f"   J_peak:        {self.J_peak:.3e} A/m²")
        print(f"   B_rem:         {self.B_rem} T")
        print(f"   Polynomial:   P{self.polynomial_degree}")
        print(f"   Timestep:      {self.dt*1000:.2f} ms")
        print(f"   Steps:         {self.num_steps}")
        print(f"   Duration:      {self.T_end*1000:.1f} ms ({self.T_end*self.frequency:.0f} periods)")
        if self.write_every_timestep:
            print(f"   Output:        every timestep")
        else:
            print(f"   Output:        {self.output_num_timestamps} timestamps")

# --- Cell tags (must match mesh.py physical groups) ---

class DomainTags:
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

    COIL_AP, COIL_AM = COIL_0, COIL_3
    COIL_BP, COIL_BM = COIL_1, COIL_4
    COIL_CP, COIL_CM = COIL_2, COIL_5


# --- Solver ---

class MaxwellSolver2D:

    def __init__(self, mesh_file="mesh.msh", config=None):
        self.config = config or SimulationConfig()
        self.tags = DomainTags()
        self.mesh_file = mesh_file
        self.mesh = None
        self.ct = None
        self.ft = None
        self.W = None
        self.w_prev = None
        self.w_sol = None
        self.sigma = None
        self.nu = None
        self.J_z = None
        self.M_x = None
        self.M_y = None

    def load_mesh(self):
        print("\n📖 Loading mesh...")
        mesh_path = Path(self.mesh_file).expanduser()
        if not mesh_path.is_absolute():
            mesh_path = (Path.cwd() / mesh_path).resolve()
        if not mesh_path.exists():
            raise FileNotFoundError(
                f"Mesh file not found: {mesh_path}\n"
                "Generate it first with: python mesh.py"
            )
        self.mesh, self.ct, *rest = gmshio.read_from_msh(
            str(mesh_path), MPI.COMM_WORLD, rank=0, gdim=2
        )
        self.ft = rest[0] if rest else None
        print(f"   ✅ {self.ct.values.size} cells loaded")
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.ct)
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.ft) if self.ft else ufl.ds
        
    def setup_materials(self):
        print("\n🔧 Setting up materials...")
        
        DG0 = fem.functionspace(self.mesh, ("DG", 0))
        mu0 = self.config.mu0
        
        sigma_vals = {
            self.tags.OUTER_AIR: 0.0,
            self.tags.AIRGAP_INNER: 0.0,
            self.tags.AIRGAP_OUTER: 0.0,
            self.tags.ROTOR: 2e6,
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
        
        nu_vals = {
            self.tags.OUTER_AIR: 1/mu0,
            self.tags.AIRGAP_INNER: 1/mu0,
            self.tags.AIRGAP_OUTER: 1/mu0,
            self.tags.ROTOR: 1/(mu0*1000),
            self.tags.PM_N: 1/mu0,
            self.tags.PM_S: 1/mu0,
            self.tags.STATOR: 1/(mu0*1000),
            self.tags.COIL_AP: 1/(mu0*0.999991),
            self.tags.COIL_AM: 1/(mu0*0.999991),
            self.tags.COIL_BP: 1/(mu0*0.999991),
            self.tags.COIL_BM: 1/(mu0*0.999991),
            self.tags.COIL_CP: 1/(mu0*0.999991),
            self.tags.COIL_CM: 1/(mu0*0.999991),
        }
        
        self.sigma = fem.Function(DG0)
        self.nu = fem.Function(DG0)
        
        for tag in sigma_vals:
            cells = self.ct.find(tag)
            if len(cells):
                self.sigma.x.array[cells] = sigma_vals[tag]
                self.nu.x.array[cells] = nu_vals[tag]
        
        print("   ✅ Materials assigned (σ, ν)")
        
    def setup_function_spaces(self):
        print("\n🎯 Setting up function spaces...")
        deg = self.config.polynomial_degree
        PA = basix.ufl.element("Lagrange", self.mesh.basix_cell(), deg)
        PV = basix.ufl.element("Lagrange", self.mesh.basix_cell(), deg)
        self.W = fem.functionspace(
            self.mesh, basix.ufl.mixed_element([PA, PV])
        )
        
        print(f"   ✅ Mixed DOFs (P{deg}): {self.W.dofmap.index_map.size_global * self.W.dofmap.index_map_bs}")
        self.w_prev = fem.Function(self.W)
        self.w_sol = fem.Function(self.W)
        
    def setup_boundary_conditions(self):
        print("\n🔒 Setting up boundary conditions...")
        tdim = self.mesh.topology.dim
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
        W1, _ = self.W.sub(1).collapse()
        self.mesh.topology.create_connectivity(tdim, 0)
        self.mesh.topology.create_connectivity(0, tdim)
        c2v = self.mesh.topology.connectivity(tdim, 0)
        
        rotor_cells = self.ct.find(self.tags.ROTOR)
        assert rotor_cells.size > 0, "No rotor cells found for V grounding"
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
        print("\n⚡ Initializing sources...")
        DG0 = fem.functionspace(self.mesh, ("DG", 0))
        self.J_z = fem.Function(DG0)
        print(f"   ✅ Current density: J_peak = {self.config.J_peak:.3e} A/m²")
        self.M_x = fem.Function(DG0)
        self.M_y = fem.Function(DG0)
        for pm_tag, sign in [(self.tags.PM_N, +1), (self.tags.PM_S, -1)]:
            cells = self.ct.find(pm_tag)
            if cells.size == 0:
                continue
            
            for c in cells:
                cell_geom_dofs = self.mesh.geometry.dofmap[c]
                cx = np.mean(self.mesh.geometry.x[cell_geom_dofs, 0])
                cy = np.mean(self.mesh.geometry.x[cell_geom_dofs, 1])
                theta = np.arctan2(cy, cx)
                if theta < 0:
                    theta += 2 * np.pi
                pole_idx = int(np.round(theta / self.config.pole_angle_step))
                theta_pole_center = (pole_idx + 0.5) * self.config.pole_angle_step
                self.M_x.x.array[c] = sign * self.config.M_rem * np.cos(theta_pole_center)
                self.M_y.x.array[c] = sign * self.config.M_rem * np.sin(theta_pole_center)
        
        print(f"   ✅ Magnetization: B_rem = {self.config.B_rem} T, {self.config.n_poles} poles")
        
    def create_variational_form(self, verbose=True):
        # Weak form: rotor σ,ω; PM −M·curl(v); coils J_z; tiny ds term for stability.
        if verbose:
            print("\n📐 Creating variational form...")
        
        def dxc(tags):
            if not tags:
                return self.dx(999)
            m = self.dx(tags[0])
            for t in tags[1:]:
                m += self.dx(t)
            return m
        
        Omega_conducting = [self.tags.ROTOR]
        Omega_coils = [
            self.tags.COIL_AP, self.tags.COIL_AM,
            self.tags.COIL_BP, self.tags.COIL_BM,
            self.tags.COIL_CP, self.tags.COIL_CM
        ]
        Omega_pm = [self.tags.PM_N, self.tags.PM_S]
        (Az, V) = ufl.TrialFunctions(self.W)
        (v, q) = ufl.TestFunctions(self.W)
        Az_prev, _ = ufl.split(self.w_prev)
        xcoord = ufl.SpatialCoordinate(self.mesh)
        x, y = xcoord[0], xcoord[1]
        omega = fem.Constant(self.mesh, self.config.omega_m)
        dt = fem.Constant(self.mesh, self.config.dt)
        u_rot_x = -omega * y
        u_rot_y = omega * x
        epsV = fem.Constant(self.mesh, 1e-12)
        
        a = self.nu * ufl.inner(ufl.grad(Az), ufl.grad(v)) * self.dx
        a += self.sigma * (u_rot_x*ufl.grad(Az)[0] + u_rot_y*ufl.grad(Az)[1]) * v * dxc(Omega_conducting)
        a += (self.sigma/dt) * Az * v * dxc(Omega_conducting)
        a += self.sigma * ufl.inner(ufl.grad(V), ufl.grad(q)) * dxc(Omega_conducting)
        a += (self.sigma/dt) * Az * q * dxc(Omega_conducting)
        a += epsV * V * q * dxc(Omega_conducting)
        n = ufl.FacetNormal(self.mesh)
        a += (1e-16) * v * (n[0]*Az.dx(0) - n[1]*Az.dx(1)) * self.ds
        
        L = (self.sigma/dt) * Az_prev * v * dxc(Omega_conducting)
        for coil in Omega_coils:
            L += self.J_z * v * self.dx(coil)
        curl_v = ufl.as_vector((v.dx(1), -v.dx(0)))
        M_vec = ufl.as_vector((self.M_x, self.M_y))
        L += -ufl.inner(M_vec, curl_v) * dxc(Omega_pm)
        vXB_prev = omega * (y*ufl.grad(Az_prev)[0] - x*ufl.grad(Az_prev)[1])
        L += (self.sigma/dt) * Az_prev * q * dxc(Omega_conducting)
        L += -self.sigma * vXB_prev * q * dxc(Omega_conducting)
        
        if verbose:
            print("   ✅ Variational form created")
        
        self.a = a
        self.L = L
        
    def update_currents(self, t):
        omega_e = self.config.omega_e
        J_peak = self.config.J_peak
        if omega_e == 0.0 or self.config.frequency == 0.0:
            IA = 0.0
            IB = 0.0
            IC = 0.0
        else:
            IA = J_peak * np.sin(omega_e * t)
            IB = J_peak * np.sin(omega_e * t - 2*np.pi/3)
            IC = J_peak * np.sin(omega_e * t + 2*np.pi/3)
        for c in self.ct.find(self.tags.COIL_AP): self.J_z.x.array[c] = IA
        for c in self.ct.find(self.tags.COIL_AM): self.J_z.x.array[c] = -IA
        for c in self.ct.find(self.tags.COIL_BP): self.J_z.x.array[c] = IB
        for c in self.ct.find(self.tags.COIL_BM): self.J_z.x.array[c] = -IB
        for c in self.ct.find(self.tags.COIL_CP): self.J_z.x.array[c] = IC
        for c in self.ct.find(self.tags.COIL_CM): self.J_z.x.array[c] = -IC
        
    def rotate_magnetization(self, t):
        theta_rot = self.config.omega_m * t
        
        for pm_tag, sign in [(self.tags.PM_N, +1), (self.tags.PM_S, -1)]:
            cells = self.ct.find(pm_tag)
            for c in cells:
                cell_geom_dofs = self.mesh.geometry.dofmap[c]
                cx = np.mean(self.mesh.geometry.x[cell_geom_dofs, 0])
                cy = np.mean(self.mesh.geometry.x[cell_geom_dofs, 1])
                theta = np.arctan2(cy, cx)
                if theta < 0:
                    theta += 2 * np.pi
                pole_idx = int(np.round(theta / self.config.pole_angle_step))
                theta_pole_center = (pole_idx + 0.5) * self.config.pole_angle_step
                theta_now = theta_pole_center + theta_rot
                self.M_x.x.array[c] = sign * self.config.M_rem * np.cos(theta_now)
                self.M_y.x.array[c] = sign * self.config.M_rem * np.sin(theta_now)
        self.M_x.x.scatter_forward()
        self.M_y.x.scatter_forward()

    def _final_brms_report(self, Az_sol):
        # Domain and air-gap B_rms from Az (diagnostic).
        Bx = ufl.grad(Az_sol)[1]
        By = -ufl.grad(Az_sol)[0]
        b2 = Bx * Bx + By * By
        area_all = fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=self.mesh)))
        val_all = fem.assemble_scalar(fem.form(b2 * ufl.dx(domain=self.mesh)))
        Brms_all = float(np.sqrt(val_all / max(area_all, 1e-18)))

        def _cell_r(c: int) -> float:
            g = self.mesh.geometry.dofmap[c]
            cx = float(np.mean(self.mesh.geometry.x[g, 0]))
            cy = float(np.mean(self.mesh.geometry.x[g, 1]))
            return math.hypot(cx, cy)

        pm_cells = np.concatenate([self.ct.find(self.tags.PM_N), self.ct.find(self.tags.PM_S)])
        stator_cells = self.ct.find(self.tags.STATOR)
        if pm_cells.size and stator_cells.size:
            rin = max(_cell_r(int(c)) for c in pm_cells)
            rout = min(_cell_r(int(c)) for c in stator_cells)
        else:
            rin, rout = 0.0, 1.0
        xcoord = ufl.SpatialCoordinate(self.mesh)
        r = ufl.sqrt(xcoord[0] ** 2 + xcoord[1] ** 2)
        mask = ufl.conditional(
            ufl.lt(r, rout), ufl.conditional(ufl.gt(r, rin), 1.0, 0.0), 0.0
        )
        area_gap = fem.assemble_scalar(fem.form(mask * ufl.dx(domain=self.mesh)))
        val_gap = fem.assemble_scalar(fem.form(b2 * mask * ufl.dx(domain=self.mesh)))
        Brms_gap = float(np.sqrt(val_gap / max(area_gap, 1e-18)))
        print(f"   Final B_rms (domain) ≈ {Brms_all:.3e} T, B_rms (air-gap) ≈ {Brms_gap:.3e} T")

    def solve(self, output_file=None, target_times_ms=None):
        # Time integration, XDMF output, optional target times (ms), air-gap B diagnostics.
        print("\n⏱️  Starting time-stepping simulation...")
        if target_times_ms is not None:
            target_times = [t_ms / 1000.0 for t_ms in target_times_ms]
            target_times = sorted(target_times)
            T_end = max(target_times)
            print(f"   Target times: {[f'{t*1000:.1f}' for t in target_times]} ms")
            print(f"   Duration: {T_end*1000:.1f} ms")
        else:
            target_times = None
            num_steps = self.config.num_steps
            T_end = self.config.T_end
            print(f"   Steps: {num_steps}")
            print(f"   Duration: {T_end*1000:.1f} ms")
        print()
        if output_file is None:
            output_file = Path(__file__).resolve().parent / "result.xdmf"
        output_path = Path(output_file).expanduser()
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            out = io.XDMFFile(self.mesh.comm, str(output_path), "w")
        except RuntimeError as exc:
            if "HDF5" not in str(exc) and "hdf5" not in str(exc):
                raise
            alt_path = output_path.with_name(
                f"{output_path.stem}_{int(time.time())}{output_path.suffix}"
            )
            print(
                f"   ⚠️ Output file locked, using alternate output: {alt_path.name}"
            )
            output_path = alt_path
            out = io.XDMFFile(self.mesh.comm, str(output_path), "w")
        out.write_mesh(self.mesh)
        if self.ct is not None:
            try:
                out.write_meshtags(self.ct)
            except Exception:
                pass
        deg = self.config.polynomial_degree
        V_B = fem.functionspace(self.mesh, ("Lagrange", deg, (2,)))
        self.B_field = fem.Function(V_B)
        self.B_field.name = "B"
        V_B_mag = fem.functionspace(self.mesh, ("Lagrange", deg))
        self.B_mag = fem.Function(V_B_mag)
        self.B_mag.name = "B_mag"
        petsc_options = {
            "ksp_type": "gmres",
            "ksp_rtol": 1e-12,
            "ksp_error_if_not_converged": True,
            "pc_type": "lu",
            "pc_factor_shift_type": "NONZERO",
        }

        def _cell_r(c: int) -> float:
            g = self.mesh.geometry.dofmap[c]
            cx = float(np.mean(self.mesh.geometry.x[g, 0]))
            cy = float(np.mean(self.mesh.geometry.x[g, 1]))
            return math.hypot(cx, cy)

        pm_cells = np.concatenate([self.ct.find(self.tags.PM_N), self.ct.find(self.tags.PM_S)])
        stator_cells = self.ct.find(self.tags.STATOR)
        if pm_cells.size and stator_cells.size:
            rin = max(_cell_r(int(c)) for c in pm_cells)
            rout = min(_cell_r(int(c)) for c in stator_cells)
        else:
            rin, rout = 0.0, 1.0

        xcoord = ufl.SpatialCoordinate(self.mesh)
        r = ufl.sqrt(xcoord[0] ** 2 + xcoord[1] ** 2)
        gap_mask = ufl.conditional(
            ufl.lt(r, rout),
            ufl.conditional(ufl.gt(r, rin), 1.0, 0.0),
            0.0,
        )
        area_gap = fem.assemble_scalar(fem.form(gap_mask * ufl.dx(domain=self.mesh)))

        peak_gap_brms = -1.0
        peak_gap_time = 0.0

        def _solve_with_residual_trace(prob: LinearProblem, stage_label: str):
            ksp = prob.solver

            def _monitor(_ksp, its, rnorm):
                print(f"      [{stage_label}] iter {its:3d}: |r|={float(rnorm):.6e}")

            ksp.setMonitor(_monitor)
            sol = prob.solve()
            try:
                ksp.cancelMonitor()
            except Exception:
                pass
            return sol, ksp

        start_time = time.time()
        t = 0.0
        
        if target_times is not None:
            # Step to each target time
            num_steps = len(target_times)
            step = 0
            
            # Initialize solution at t=0 (needed for time-stepping)
            self.update_currents(0.0)
            self.rotate_magnetization(0.0)
            self.create_variational_form(verbose=False)
            prob = LinearProblem(
                self.a, self.L, bcs=self.bcs,
                petsc_options=petsc_options,
                petsc_options_prefix="mixed"
            )
            self.w_sol, ksp = _solve_with_residual_trace(prob, "init t=0.00 ms")
            ksp_its = int(ksp.getIterationNumber())
            ksp_res = float(ksp.getResidualNorm())
            ksp_reason = int(ksp.getConvergedReason())
            Az_sol, _ = self.w_sol.split()
            
            # Compute B field
            Bx_expr = ufl.grad(Az_sol)[1]
            By_expr = -ufl.grad(Az_sol)[0]
            B_expr = ufl.as_vector((Bx_expr, By_expr))
            Bvec_interp = fem.Expression(B_expr, V_B.element.interpolation_points)
            self.B_field.interpolate(Bvec_interp)
            self.B_field.x.scatter_forward()
            
            # Save initial state if t=0 is in target times
            if 0.0 in target_times:
                Az_sol.name = "Az"
                out.write_function(Az_sol, 0.0)
                Bmag_expr = fem.Expression(
                    ufl.sqrt(Bx_expr * Bx_expr + By_expr * By_expr),
                    V_B_mag.element.interpolation_points,
                )
                self.B_mag.interpolate(Bmag_expr)
                self.B_mag.x.scatter_forward()
                out.write_function(self.B_field, 0.0)
                out.write_function(self.B_mag, 0.0)
                norm_Az = np.linalg.norm(Az_sol.x.array)
                print(f"   Step {step:3d}/{num_steps}  t={0.0*1e3:5.2f} ms  ||Az||={norm_Az:.2e}")
                step += 1
            
            # Update previous solution for time-stepping
            self.w_prev.x.array[:] = self.w_sol.x.array[:]
            
            # Step to each target time
            for target_t in target_times:
                if target_t == 0.0:
                    continue  # Already handled
                
                # Step from current time to target time
                while t < target_t - 1e-10:  # Small tolerance for floating point
                    dt_step = min(self.config.dt, target_t - t)
                    t += dt_step
                    
                    # Update sources
                    self.update_currents(t)
                    self.rotate_magnetization(t)
                    
                    # Recreate linear form L with updated sources
                    self.create_variational_form(verbose=False)
                    
                    # Solve linear system
                    prob = LinearProblem(
                        self.a, self.L, bcs=self.bcs,
                        petsc_options=petsc_options,
                        petsc_options_prefix="mixed"
                    )
                    self.w_sol, ksp = _solve_with_residual_trace(
                        prob, f"t={t*1e3:.2f} ms"
                    )
                    ksp_its = int(ksp.getIterationNumber())
                    ksp_res = float(ksp.getResidualNorm())
                    ksp_reason = int(ksp.getConvergedReason())
                    
                    # Extract components
                    Az_sol, _ = self.w_sol.split()
                    
                    # Compute B field
                    Bx_expr = ufl.grad(Az_sol)[1]
                    By_expr = -ufl.grad(Az_sol)[0]
                    B_expr = ufl.as_vector((Bx_expr, By_expr))
                    Bvec_interp = fem.Expression(B_expr, V_B.element.interpolation_points)
                    self.B_field.interpolate(Bvec_interp)
                    self.B_field.x.scatter_forward()
                    
                    # Update previous solution
                    self.w_prev.x.array[:] = self.w_sol.x.array[:]
                
                # Save at target time
                Az_sol.name = "Az"
                out.write_function(Az_sol, t)
                Bmag_expr = fem.Expression(
                    ufl.sqrt(Bx_expr * Bx_expr + By_expr * By_expr),
                    V_B_mag.element.interpolation_points,
                )
                self.B_mag.interpolate(Bmag_expr)
                self.B_mag.x.scatter_forward()
                out.write_function(self.B_field, t)
                out.write_function(self.B_mag, t)
                norm_Az = np.linalg.norm(Az_sol.x.array)
                # Per-step B_rms in air-gap, and track peak over all saved steps.
                Bx = ufl.grad(Az_sol)[1]
                By = -ufl.grad(Az_sol)[0]
                b2 = Bx * Bx + By * By
                val_gap = fem.assemble_scalar(fem.form(b2 * gap_mask * ufl.dx(domain=self.mesh)))
                Brms_gap = float(np.sqrt(val_gap / max(area_gap, 1e-18)))
                if Brms_gap > peak_gap_brms:
                    peak_gap_brms = Brms_gap
                    peak_gap_time = t

                print(
                    f"   Step {step:3d}/{num_steps}  t={t*1e3:5.2f} ms  "
                    f"||Az||={norm_Az:.2e}  KSP it={ksp_its:3d}  "
                    f"|r|={ksp_res:.3e}  reason={ksp_reason}  "
                    f"B_rms(gap)={Brms_gap:.3e} T"
                )
                step += 1
        else:
            # Original uniform stepping
            num_steps = self.config.num_steps
            if self.config.write_every_timestep:
                write_steps = set(range(1, num_steps + 1))
            else:
                n_out = max(1, min(num_steps, int(self.config.output_num_timestamps)))
                write_steps = set(np.linspace(1, num_steps, n_out, dtype=int).tolist())
                write_steps.add(num_steps)
            for step in range(1, num_steps + 1):
                t += self.config.dt
                
                # Update sources
                self.update_currents(t)
                self.rotate_magnetization(t)
                
                # Recreate linear form L with updated sources (M_x, M_y, J_z)
                # This is necessary because PM rotation changes M_x, M_y over time
                # Use verbose=False to avoid printing every step
                self.create_variational_form(verbose=False)
                
                # Solve linear system
                prob = LinearProblem(
                    self.a, self.L, bcs=self.bcs,
                    petsc_options=petsc_options,
                    petsc_options_prefix="mixed"
                )
                self.w_sol, ksp = _solve_with_residual_trace(
                    prob, f"t={t*1e3:.2f} ms"
                )
                ksp_its = int(ksp.getIterationNumber())
                ksp_res = float(ksp.getResidualNorm())
                ksp_reason = int(ksp.getConvergedReason())
                
                # Extract components
                Az_sol, _ = self.w_sol.split()
                
                # Compute B field: B = curl(A) = (dAz/dy, -dAz/dx)
                # Use projection for better accuracy
                Bx_expr = ufl.grad(Az_sol)[1]
                By_expr = -ufl.grad(Az_sol)[0]
                B_expr = ufl.as_vector((Bx_expr, By_expr))
                Bvec_interp = fem.Expression(B_expr, V_B.element.interpolation_points)
                self.B_field.interpolate(Bvec_interp)
                self.B_field.x.scatter_forward()
                
                # Update previous solution
                self.w_prev.x.array[:] = self.w_sol.x.array[:]
                
                # Save output based on configured timestamp policy.
                if step in write_steps:
                    # Write A and B fields directly for robust visualization.
                    Az_sol.name = "Az"
                    out.write_function(Az_sol, t)
                    Bmag_expr = fem.Expression(
                        ufl.sqrt(Bx_expr * Bx_expr + By_expr * By_expr),
                        V_B_mag.element.interpolation_points,
                    )
                    self.B_mag.interpolate(Bmag_expr)
                    self.B_mag.x.scatter_forward()
                    out.write_function(self.B_field, t)
                    out.write_function(self.B_mag, t)
                
                # Progress
                norm_Az = np.linalg.norm(Az_sol.x.array)
                Bx = ufl.grad(Az_sol)[1]
                By = -ufl.grad(Az_sol)[0]
                b2 = Bx * Bx + By * By
                val_gap = fem.assemble_scalar(fem.form(b2 * gap_mask * ufl.dx(domain=self.mesh)))
                Brms_gap = float(np.sqrt(val_gap / max(area_gap, 1e-18)))
                if Brms_gap > peak_gap_brms:
                    peak_gap_brms = Brms_gap
                    peak_gap_time = t
                print(
                    f"   Step {step:3d}/{num_steps}  t={t*1e3:5.2f} ms  "
                    f"||Az||={norm_Az:.2e}  KSP it={ksp_its:3d}  "
                    f"|r|={ksp_res:.3e}  reason={ksp_reason}  "
                    f"B_rms(gap)={Brms_gap:.3e} T"
                )
                if step == num_steps:
                    self._final_brms_report(Az_sol)

        if target_times is not None and step > 0:
            self._final_brms_report(Az_sol)
        
        out.close()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Simulation complete in {elapsed:.1f}s")
        print(f"   Output: {output_path}")
        if peak_gap_brms >= 0.0:
            print(
                f"   Peak air-gap B_rms: {peak_gap_brms:.3e} T "
                f"at t={peak_gap_time*1e3:.2f} ms"
            )
        
    def run(self):
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


if __name__ == "__main__":
    default_mesh = Path(__file__).resolve().parent / "mesh.msh"
    solver = MaxwellSolver2D(mesh_file=str(default_mesh))
    solver.run()