#!/usr/bin/env python3
"""
3D Maxwell A-V Mixed Formulation Solver (Refactored)
====================================================

This module provides a modular 3D A-V solver for the PMSM mesh.
The code has been refactored into 3 main modules:

- setup: Configuration, domain metadata, mesh loading, materials, BCs
- solver: Form building, matrix assembly, solver config, source management
- postprocess: B-field computation
"""

from __future__ import annotations

import csv
from typing import Sequence

import basix.ufl
import numpy as np
from dolfinx import fem, io
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from postprocess import compute_B_field
from setup import (
    DomainTags3D,
    SimulationConfig3D,
    load_mesh,
    maybe_retag_cells,
    setup_boundary_conditions,
    setup_materials,
)
from solver import (
    assemble_system_matrix,
    build_forms,
    configure_solver,
    current_stats,
    initialise_magnetisation,
    rebuild_linear_forms,
    rotate_magnetization,
    setup_sources,
    update_currents,
)


class MaxwellAVSolver3D:
    """3D A-V mixed formulation driver."""

    def __init__(self, config: SimulationConfig3D | None = None) -> None:
        self.config = config or SimulationConfig3D()
        self.mesh = None
        self.ct = None
        self.ft = None

        self.dx = None
        self.ds = None

        self.sigma = None
        self.nu = None
        self.density = None

        self.J_z = None
        self.M_vec = None

        self.A_space = None
        self.V_space = None
        self.B_space = None  # Lagrange space for B = curl(A)
        self.B_magnitude_space = None
        self.A_prev = None

        self.bc_A = None
        self.bc_V = None
        self.block_bcs = None

        self.a_blocks = None
        self.L_blocks = None
        self.mat_blocks = None
        self.mat_nest = None
        self.A00_standalone = None
        self._rhs_blocks = None
        self.ksp = None

        self.dx_conductors = None
        self.dx_magnets = None

    def setup(self) -> None:
        """High-level setup routine."""
        if self.mesh is None or self.mesh.comm.rank == 0:
            print("\n" + "="*70)
            print("🔧 SOLVER SETUP")
            print("="*70)
        
        if self.mesh is None or self.mesh.comm.rank == 0:
            print("\n[1/7] Loading mesh...")
        self._load_mesh()
        
        if self.mesh.comm.rank == 0:
            print("[2/7] Setting up measures...")
        self._setup_measures()
        
        if self.mesh.comm.rank == 0:
            print("[3/7] Setting up materials...")
        self._setup_materials()
        
        if self.mesh.comm.rank == 0:
            print("[4/7] Setting up function spaces...")
        self._setup_function_spaces()
        
        if self.mesh.comm.rank == 0:
            print("[5/7] Setting up boundary conditions...")
        self._setup_boundary_conditions()
        
        if self.mesh.comm.rank == 0:
            print("[6/7] Setting up sources (currents & magnets)...")
        self._setup_sources()
        
        if self.mesh.comm.rank == 0:
            print("[7/7] Building forms and assembling system matrix...")
        self._build_forms()
        self._assemble_system_matrix()
        
        if self.mesh.comm.rank == 0:
            print("\n✅ Setup complete!")
            print("="*70 + "\n")

    def _load_mesh(self) -> None:
        """Read XDMF mesh + tags exported by mesh_3D.py."""
        self.mesh, self.ct, self.ft = load_mesh(self.config.mesh_path)
        self.ct = maybe_retag_cells(self.mesh, self.ct)

    def _setup_measures(self) -> None:
        """Set up UFL measures."""
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.ct)
        if self.ft is not None:
            self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.ft)
        else:
            self.ds = ufl.ds(domain=self.mesh)
        self.dx_conductors = self._measure_over(DomainTags3D.conducting())
        self.dx_magnets = self._measure_over(DomainTags3D.MAGNETS)

    def _setup_materials(self) -> None:
        """Populate piecewise-constant fields (σ, ν, ρ)."""
        self.sigma, self.nu, self.density = setup_materials(
            self.mesh, self.ct, self.config
        )

    def _setup_function_spaces(self) -> None:
        """Define curl-conforming vector space and scalar potential space."""
        tdim = self.mesh.topology.dim
        if tdim != 3:
            raise RuntimeError("This solver expects a 3D mesh.")

        nedelec = basix.ufl.element(
            "N1curl", self.mesh.basix_cell(), self.config.degree_A
        )
        lagrange = basix.ufl.element(
            "Lagrange", self.mesh.basix_cell(), self.config.degree_V
        )

        self.A_space = fem.functionspace(self.mesh, nedelec)
        self.V_space = fem.functionspace(self.mesh, lagrange)
        # B = curl(A) space: Lagrange vector for visualization/diagnostics
        lagrange_vec = basix.ufl.element(
            "Lagrange", self.mesh.basix_cell(), self.config.degree_A, shape=(3,)
        )
        self.B_space = fem.functionspace(self.mesh, lagrange_vec)
        # B_magnitude space: scalar Lagrange for color mapping in ParaView
        self.B_magnitude_space = fem.functionspace(self.mesh, lagrange)
        self.A_prev = fem.Function(self.A_space, name="A_prev")

        ndofs = (
            self.A_space.dofmap.index_map.size_global * self.A_space.dofmap.index_map_bs
        )
        if self.mesh.comm.rank == 0:
            print(f"🎯 DOFs(A): {ndofs}, DOFs(V): {self.V_space.dofmap.index_map.size_global}, DOFs(B): {self.B_space.dofmap.index_map.size_global * 3}")

    def _setup_boundary_conditions(self) -> None:
        """Impose A = 0 and V = 0 on the exterior surface."""
        self.bc_A, self.bc_V, self.block_bcs = setup_boundary_conditions(
            self.mesh, self.ft, self.A_space, self.V_space
        )

    def _setup_sources(self) -> None:
        """Allocate coil current density & magnetisation fields."""
        self.J_z, self.M_vec = setup_sources(self.mesh, self.ct)
        initialise_magnetisation(self.mesh, self.ct, self.M_vec, self.config)

    def _build_forms(self) -> None:
        """Create block bilinear/linear forms."""
        self.a_blocks, self.L_blocks = build_forms(
            self.mesh,
            self.A_space,
            self.V_space,
            self.sigma,
            self.nu,
            self.J_z,
            self.M_vec,
            self.A_prev,
            self.dx,
            self.dx_conductors,
            self.dx_magnets,
            self.config,
        )

    def _assemble_system_matrix(self) -> None:
        """Create (and keep) the nested PETSc operator."""
        self.mat_blocks, self.mat_nest, self.A00_standalone = assemble_system_matrix(
            self.mesh, self.a_blocks, self.block_bcs
        )

    def update_currents(self, t: float) -> None:
        """Populate J_z for each coil using three-phase mapping."""
        update_currents(self.mesh, self.ct, self.J_z, self.config, t)

    def rotate_magnetization(self, t: float) -> None:
        """Rotate PM magnetization with rotor."""
        rotate_magnetization(self.mesh, self.ct, self.M_vec, self.config, t)

    def current_stats(self) -> float:
        """Return max |J_z| for diagnostics."""
        return current_stats(self.J_z)

    def assemble_rhs(self) -> PETSc.Vec:
        """Assemble nested RHS vector with the current field state.
        
        Note: For Nédélec elements, volume integrals contribute to edge DOFs.
        If edges are on the boundary, all RHS contributions go to boundary DOFs.
        We do NOT zero boundary DOFs here because:
        1. The matrix was assembled with BCs (boundary rows are identity)
        2. Zeroing RHS boundary DOFs would lose all contributions
        3. The matrix will enforce zero solution at boundary DOFs regardless of RHS
        """
        if self.L_blocks is None or self.a_blocks is None:
            raise RuntimeError("Forms not built.")

        block_vecs = []
        for i, L_form in enumerate(self.L_blocks):
            vec = petsc.create_vector(L_form)
            petsc.assemble_vector(vec, L_form)
            vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            
            # Apply lifting to account for BC coupling between blocks
            if i < len(self.a_blocks):
                for j, a_form in enumerate(self.a_blocks[i]):
                    if a_form is not None and i != j and j < len(self.block_bcs) and self.block_bcs[j]:
                        petsc.apply_lifting(vec, [a_form], bcs=[self.block_bcs[j]])
            
            vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            
            # CRITICAL FIX: Do NOT zero boundary DOFs in RHS for Nédélec elements
            # The matrix was assembled with BCs, so boundary rows are identity rows
            # that enforce zero solution. Zeroing RHS boundary DOFs would lose all
            # contributions since they all go to boundary DOFs for Nédélec elements.
            # 
            # For zero BCs with Nédélec elements, we skip set_bc() on the RHS.
            # The matrix will enforce the boundary condition in the solution.
            #
            # Only apply set_bc for non-zero BCs or for Lagrange elements (V block)
            if i == 1 and i < len(self.block_bcs) and self.block_bcs[i]:
                # V block (Lagrange) - safe to apply BCs
                petsc.set_bc(vec, self.block_bcs[i])
            # A block (Nédélec) - skip set_bc to preserve RHS contributions
            
            block_vecs.append(vec)

        self._rhs_blocks = block_vecs
        b = PETSc.Vec().createNest(block_vecs, comm=self.mesh.comm)
        return b

    def configure_solver(self) -> None:
        """Configure KSP and preconditioner."""
        self.ksp = configure_solver(self.mesh, self.mat_nest, self.mat_blocks,
                                   self.A_space, self.V_space, self.config.degree_A)

    def solve_step(self, t: float) -> tuple[fem.Function, fem.Function, int, float, float]:
        """Solve one implicit time step at time ``t``."""
        if self.ksp is None:
            if self.mesh.comm.rank == 0:
                print("⚙️  Configuring solver (first call)...")
            self.configure_solver()
        if self.L_blocks is None:
            raise RuntimeError("Linear forms not built.")

        if self.mesh.comm.rank == 0:
            print(f"\n⏱️  Solving at t = {t*1e3:.3f} ms")
            print("   [1/4] Updating sources (currents & magnetization)...")
        
        # Update sources: currents and PM magnetization (rotation)
        self.update_currents(t)
        self.rotate_magnetization(t)
        
        if self.mesh.comm.rank == 0:
            print("   [2/4] Rebuilding linear forms...")
        
        # Rebuild only linear forms because M_vec and J_z have changed
        self.L_blocks = rebuild_linear_forms(
            self.mesh,
            self.A_space,
            self.V_space,
            self.sigma,
            self.J_z,
            self.M_vec,
            self.A_prev,
            self.dx,
            self.dx_magnets,
            self.config,
        )
        
        max_J = self.current_stats()
        
        if self.mesh.comm.rank == 0:
            print("   [3/4] Assembling RHS vector...")
        
        rhs = self.assemble_rhs()
        sol_blocks = [
            petsc.create_vector(self.L_blocks[0]),
            petsc.create_vector(self.L_blocks[1]),
        ]
        sol = PETSc.Vec().createNest(sol_blocks, comm=self.mesh.comm)
        
        if self.mesh.comm.rank == 0:
            print("   [4/4] Solving linear system...")
        
        self.ksp.solve(rhs, sol)

        A_sol = fem.Function(self.A_space, name="A")
        V_sol = fem.Function(self.V_space, name="V")
        self._copy_block_to_function(sol.getNestSubVecs()[0], A_sol)
        self._copy_block_to_function(sol.getNestSubVecs()[1], V_sol)

        self.A_prev.x.array[:] = A_sol.x.array[:]
        iterations = self.ksp.getIterationNumber()
        residual = self.ksp.getResidualNorm()
        
        if self.mesh.comm.rank == 0:
            print(f"      ✓ Solve complete: {iterations} iterations, residual = {residual:.2e}")

        return A_sol, V_sol, iterations, max_J, residual

    def compute_B_field(self, A_sol: fem.Function, debug: bool = False):
        """Compute B = curl(A) and return B field, max|B|, min|B|, and ||B||."""
        return compute_B_field(
            self.mesh,
            A_sol,
            self.B_space,
            self.B_magnitude_space,
            self.config,
            cell_tags=self.ct,
            debug=debug,
        )

    def run_time_loop(self, steps: int | None = None, *, write_results: bool | None = None) -> None:
        """Execute multiple implicit steps and optionally write XDMF output."""
        num_steps = steps or self.config.num_steps
        dt = self.config.dt
        current_time = 0.0

        if self.mesh.comm.rank == 0:
            print("\n" + "="*70)
            print("🚀 STARTING TIME LOOP")
            print("="*70)
            print(f"   Steps: {num_steps}")
            print(f"   Time step: {dt*1e3:.3f} ms")
            print(f"   Total time: {num_steps*dt*1e3:.3f} ms")
            print("="*70 + "\n")

        should_write = self.config.write_results if write_results is None else write_results
        results_path = self.config.results_path
        writer = None
        
        try:
            if should_write:
                results_path.parent.mkdir(parents=True, exist_ok=True)
                if self.mesh.comm.rank == 0:
                    print(f"📝 Writing results to: {results_path}")
                writer = io.XDMFFile(MPI.COMM_WORLD, str(results_path), "w")
                
                # Write mesh geometry (CRITICAL: must be written first)
                writer.write_mesh(self.mesh)
                
                # Write cell markers (domains) - for domain visualization
                if self.ct is not None:
                    writer.write_meshtags(self.ct, self.mesh.geometry)
                    
                    # Create cell tag function for ParaView domain coloring
                    DG0 = fem.functionspace(self.mesh, ("DG", 0))
                    cell_tag_function = fem.Function(DG0, name="CellTags")
                    
                    # Map cell markers to function values
                    if self.ct.indices.size > 0:
                        cell_to_tag = {int(i): int(v) for i, v in 
                                      zip(self.ct.indices, self.ct.values)}
                        for cell_idx, tag in cell_to_tag.items():
                            if cell_idx < cell_tag_function.x.array.size:
                                cell_tag_function.x.array[cell_idx] = float(tag)
                    
                    # Write cell tag function at time 0 (static, doesn't change)
                    writer.write_function(cell_tag_function, 0.0)
                
                # Write facet markers (boundaries) - for boundary visualization
                if self.ft is not None:
                    writer.write_meshtags(self.ft, self.mesh.geometry)

            diag_path = self.config.diagnostics_path
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diag_path, "w", newline="", encoding="utf-8") as diag_file:
                diag_writer = csv.writer(diag_file)
                diag_writer.writerow(["step", "time_ms", "norm_A", "iterations", "residual", "max_J", "max_B", "min_B", "norm_B"])

                for step in range(1, num_steps + 1):
                    current_time += dt
                    A_sol, V_sol, its, max_J, residual = self.solve_step(current_time)
                    
                    if self.mesh.comm.rank == 0:
                        print("   Computing B = curl(A)...")
                    
                    # Compute B = curl(A)
                    debug_B = (step == 1)  # Debug only first step
                    B_sol, B_magnitude_sol, max_B, min_B, norm_B = self.compute_B_field(A_sol, debug=debug_B)
                    
                    # B field filtering is already done in compute_B_field() in postprocess.py
                    # No additional clipping needed here - the intelligent filtering preserves
                    # the distribution while removing artifacts
                    
                    if writer is not None:
                        # Write A field - interpolate to Lagrange space (ParaView can't read Nédélec directly)
                        A_lag_space = fem.functionspace(self.mesh, ("Lagrange", 1, (3,)))
                        A_lag = fem.Function(A_lag_space, name="A")
                        try:
                            A_lag.interpolate(A_sol)
                        except Exception as e:
                            if self.mesh.comm.rank == 0:
                                print(f"   ⚠️  Warning: Could not interpolate A field: {e}")
                            A_lag.x.array[:] = 0.0
                        
                        # Ensure function names are set for ParaView
                        V_sol.name = "V"
                        B_sol.name = "B"
                        B_magnitude_sol.name = "B_Magnitude"
                        
                        # Write node-centered fields (for surface rendering and general use)
                        # All fields written with current_time for proper time series
                        writer.write_function(A_lag, current_time)
                        writer.write_function(V_sol, current_time)
                        writer.write_function(B_sol, current_time)
                        writer.write_function(B_magnitude_sol, current_time)
                        
                        # Create cell-centered (DG) versions for better volume rendering in ParaView
                        # ParaView volume rendering works better with cell-centered data
                        from dolfinx.fem import petsc as fem_petsc
                        DG0 = fem.functionspace(self.mesh, ("DG", 0))
                        DG0_vec = fem.functionspace(self.mesh, ("DG", 0, (3,)))
                        
                        # Project to cell-centered using L2 projection for accurate volume rendering
                        def project_to_dg(source_func, dg_space):
                            """Project a function to DG space using L2 projection."""
                            from petsc4py import PETSc
                            import ufl
                            
                            dg_func = fem.Function(dg_space)
                            v = ufl.TestFunction(dg_space)
                            u = ufl.TrialFunction(dg_space)
                            
                            a = fem.form(ufl.inner(u, v) * ufl.dx)
                            L = fem.form(ufl.inner(source_func, v) * ufl.dx)
                            
                            A = fem_petsc.assemble_matrix(a)
                            A.assemble()
                            b = fem_petsc.create_vector(L)
                            fem_petsc.assemble_vector(b, L)
                            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                            
                            ksp = PETSc.KSP().create(comm=self.mesh.comm)
                            ksp.setOperators(A)
                            ksp.setType("preonly")
                            ksp.getPC().setType("jacobi")
                            x = b.duplicate()
                            ksp.solve(b, x)
                            
                            dg_func.x.array[:] = x.array_r[:dg_func.x.array.size]
                            dg_func.x.scatter_forward()
                            
                            A.destroy()
                            b.destroy()
                            x.destroy()
                            ksp.destroy()
                            
                            return dg_func
                        
                        # Project fields to cell-centered for volume rendering
                        try:
                            A_dg = project_to_dg(A_lag, DG0_vec)
                            A_dg.name = "A_cell"
                            B_dg = project_to_dg(B_sol, DG0_vec)
                            B_dg.name = "B_cell"
                            B_mag_dg = project_to_dg(B_magnitude_sol, DG0)
                            B_mag_dg.name = "B_Magnitude_cell"
                            V_dg = project_to_dg(V_sol, DG0)
                            V_dg.name = "V_cell"
                            
                            # Write cell-centered fields (for volume rendering)
                            writer.write_function(A_dg, current_time)
                            writer.write_function(B_dg, current_time)
                            writer.write_function(B_mag_dg, current_time)
                            writer.write_function(V_dg, current_time)
                        except Exception as e:
                            # If cell-centered projection fails, continue with node-centered only
                            if self.mesh.comm.rank == 0:
                                print(f"   ⚠️  Warning: Could not create cell-centered fields: {e}")
                    
                    norm_A = np.linalg.norm(A_sol.x.array)
                    diag_writer.writerow(
                        [step, current_time * 1e3, norm_A, its, residual, max_J, max_B, min_B, norm_B]
                    )
                    
                    if self.mesh.comm.rank == 0:
                        print(f"\n   ✅ Step {step:03d}/{num_steps} complete:")
                        print(f"      Time: {current_time*1e3:7.3f} ms")
                        print(f"      ||A||: {norm_A:.3e} Wb/m")
                        print(f"      Iterations: {its}")
                        print(f"      Residual: {residual:.2e}")
                        print(f"      max|J|: {max_J:.3e} A/m²")
                        print(f"      max|B|: {max_B:.3e} T")
                        print(f"      min|B|: {min_B:.3e} T")
                        print(f"      ||B||: {norm_B:.3e}")
                        print()
        finally:
            if writer is not None:
                try:
                    writer.close()
                    if self.mesh.comm.rank == 0:
                        print(f"✅ Results written to: {results_path}")
                except Exception as e:
                    if self.mesh.comm.rank == 0:
                        print(f"⚠️  Warning: Error closing XDMF file: {e}")
        
        if self.mesh.comm.rank == 0:
            print("\n" + "="*70)
            print("✅ TIME LOOP COMPLETE")
            print("="*70 + "\n")

    def _measure_over(self, markers: Sequence[int]) -> ufl.Measure:
        """Return dx restricted to multiple cell markers."""
        measure = None
        for marker in markers:
            term = self.dx(marker)
            measure = term if measure is None else measure + term
        if measure is None:
            raise ValueError("Empty marker sequence provided.")
        return measure

    @staticmethod
    def _copy_block_to_function(block_vec: PETSc.Vec, target: fem.Function) -> None:
        """Load a PETSc block vector into a Function."""
        with block_vec.localForm() as src:
            target.x.array[:] = src.array_r[: target.x.array.size]
        target.x.scatter_forward()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 3D A-V MIXED FORMULATION SOLVER (REFACTORED)")
    print("="*70)
    print("Starting solver initialization...\n")
    
    solver = MaxwellAVSolver3D()
    solver.setup()
    
    print("\n🔧 Configuring solver...")
    solver.configure_solver()
    
    solver.run_time_loop()

