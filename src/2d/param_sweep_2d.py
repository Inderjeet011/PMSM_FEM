#!/usr/bin/env python3
"""
Parameter Sweep Script for 2D Maxwell Solver
============================================
Clean parameter sweep focusing on two key parameters:
1. omega_m variations (with all others at normal)
2. frequency variations (with all others at normal)

This results in 6 simulations total (2 parameters × 3 values each)
"""

import os
import sys
import csv
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.2d import solver_2d
MaxwellSolver2D = solver_2d.MaxwellSolver2D
SimulationConfig = solver_2d.SimulationConfig


# ============================================================================
# NORMAL/BASELINE VALUES
# ============================================================================

NORMAL_PARAMS = {
    'B_rem': 1.4,              # Normal PM strength
    'omega_m': 78.5,           # Normal rotation (~750 RPM)
    'polynomial_degree': 1,    # P1 (linear)
    'J_peak': 7.07e6,          # Normal current density
    'frequency': 50.0          # Normal 50 Hz
}


# ============================================================================
# PARAMETER VARIATIONS (zero, mid, extreme)
# ============================================================================

# Set 1: omega_m variations (keep B_rem=1.4, frequency=50, J_peak=7.07e6, P1)
OMEGA_M_VARIATIONS = {
    'zero': 0.0,      # Static (no rotation)
    'mid': 78.5,      # Normal (~750 RPM)
    'extreme': 157.0  # High (~1500 RPM)
}

# Set 2: frequency variations (keep B_rem=1.4, omega_m=78.5, J_peak=7.07e6, P1)
FREQUENCY_VARIATIONS = {
    'zero': 0.0,      # DC/static
    'mid': 50.0,      # Normal 50 Hz
    'extreme': 100.0  # High frequency
}


def generate_filename(params_dict):
    """Generate descriptive filename from parameters"""
    parts = []
    
    # B_rem
    if params_dict['B_rem'] == 0.0:
        parts.append('Brem0')
    else:
        parts.append(f"Brem{params_dict['B_rem']:.1f}")
    
    # omega_m
    if params_dict['omega_m'] == 0.0:
        parts.append('omega0')
    else:
        parts.append(f"omega{params_dict['omega_m']:.0f}")
    
    # frequency
    if params_dict['frequency'] == 0.0:
        parts.append('freq0')
    else:
        parts.append(f"freq{params_dict['frequency']:.0f}")
    
    # J_peak
    if params_dict['J_peak'] == 0.0:
        parts.append('J0')
    else:
        parts.append(f"J{params_dict['J_peak']/1e6:.2f}e6")
    
    # Polynomial degree
    parts.append(f"P{params_dict['polynomial_degree']}")
    
    return "_".join(parts)


def run_single_simulation(params_dict, mesh_file, results_dir):
    """Run a single simulation with given parameters"""
    import numpy as np
    
    # Use parameters directly from params_dict (already set correctly in generate_simulation_list)
    # No need to recompute - omega_m and frequency are kept independent
    frequency = params_dict.get('frequency', 50.0)
    omega_m = params_dict.get('omega_m', 78.5)
    pole_pairs = 2  # Fixed for this motor
    
    # Create config - run for specific time points: 0, 4, 8, 16, 20 ms
    dt = 0.002  # 2 ms timestep (used for stepping between target times)
    T_end = 0.020  # 20 ms (maximum time, but we'll use target_times instead)
    config = SimulationConfig(
        pole_pairs=pole_pairs,
        frequency=frequency,
        J_peak=params_dict['J_peak'],
        B_rem=params_dict['B_rem'],
        dt=dt,
        T_end=T_end,
        polynomial_degree=params_dict['polynomial_degree'],
        omega_m=omega_m  # Pass omega_m explicitly to allow independent control
    )
    
    # Generate filename
    filename = generate_filename(params_dict)
    output_file = os.path.join(results_dir, f"{filename}.xdmf")
    
    print("\n" + "="*80)
    print(f"Running simulation: {filename}")
    print("="*80)
    
    try:
        # Create solver
        solver = MaxwellSolver2D(mesh_file=mesh_file, config=config)
        
        # Run simulation with target times: 0, 4, 8, 16, 20 ms
        target_times_ms = [0, 4, 8, 16, 20]
        solver.load_mesh()
        solver.setup_materials()
        solver.setup_function_spaces()
        solver.setup_boundary_conditions()
        solver.initialize_sources()
        solver.create_variational_form()
        solver.solve(output_file=output_file, target_times_ms=target_times_ms)
        
        # Extract final results
        Az_sol, V_sol = solver.w_sol.split()
        norm_Az = np.linalg.norm(Az_sol.x.array)
        
        # Compute B field magnitude
        import ufl
        from dolfinx import fem
        Bx = ufl.grad(Az_sol)[1]
        By = -ufl.grad(Az_sol)[0]
        B_mag = ufl.sqrt(Bx*Bx + By*By)
        area = fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=solver.mesh)))
        B2_integral = fem.assemble_scalar(fem.form(B_mag*B_mag * ufl.dx(domain=solver.mesh)))
        B_rms = float(np.sqrt(B2_integral / max(area, 1e-18)))
        
        return {
            'filename': filename,
            'success': True,
            'norm_Az': float(norm_Az),
            'B_rms': float(B_rms),
            'error': None
        }
        
    except Exception as e:
        print(f"\n❌ ERROR in simulation {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'filename': filename,
            'success': False,
            'norm_Az': None,
            'B_rms': None,
            'error': str(e)
        }


def generate_simulation_list():
    """Generate list of simulations: clean two-parameter sweep"""
    simulations = []
    
    # Set 1: omega_m variations (keep frequency=50, B_rem=1.4, J_peak=7.07e6, P1)
    for value_name, omega_m_value in OMEGA_M_VARIATIONS.items():
        params_dict = NORMAL_PARAMS.copy()
        params_dict['omega_m'] = omega_m_value
        # Keep frequency fixed at 50 Hz (normal)
        params_dict['frequency'] = 50.0
        
        sim_name = f"omega_m_{value_name}"
        simulations.append({
            'name': sim_name,
            'params': params_dict,
            'varied_param': 'omega_m',
            'value_name': value_name
        })
    
    # Set 2: frequency variations (keep omega_m=78.5, B_rem=1.4, J_peak=7.07e6, P1)
    for value_name, frequency_value in FREQUENCY_VARIATIONS.items():
        params_dict = NORMAL_PARAMS.copy()
        params_dict['frequency'] = frequency_value
        # Keep omega_m fixed at 78.5 rad/s (normal)
        params_dict['omega_m'] = 78.5
        
        sim_name = f"frequency_{value_name}"
        simulations.append({
            'name': sim_name,
            'params': params_dict,
            'varied_param': 'frequency',
            'value_name': value_name
        })
    
    return simulations


def main():
    """Main parameter sweep execution"""
    import numpy as np
    
    # Configuration
    mesh_file = "../../meshes/2d/motor.msh"
    results_dir = "../../results/2d/param_sweep"
    summary_file = os.path.join(results_dir, "parameter_sweep_summary.csv")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*80)
    print(" PARAMETER SWEEP: 2D MAXWELL SOLVER")
    print("="*80)
    print("\n📋 Strategy: Clean two-parameter sweep")
    print(f"\n📊 Normal/Baseline parameters:")
    for param, value in NORMAL_PARAMS.items():
        print(f"   {param:20s}: {value}")
    
    print(f"\n📊 Parameter Sets:")
    print(f"   Set 1: omega_m variations (zero=0, mid=78.5, extreme=157)")
    print(f"          Keep FIXED: B_rem=1.4, frequency=50, J_peak=7.07e6, P1")
    print(f"   Set 2: frequency variations (zero=0, mid=50, extreme=100)")
    print(f"          Keep FIXED: B_rem=1.4, omega_m=78.5, J_peak=7.07e6, P1")
    
    # Generate simulation list
    simulations = generate_simulation_list()
    total_sims = len(simulations)
    
    print(f"\n📊 Total simulations: {total_sims} (2 parameters × 3 values each)")
    print(f"📁 Results directory: {results_dir}")
    print(f"📄 Summary file: {summary_file}")
    
    # Run simulations
    results = []
    for idx, sim_info in enumerate(simulations, 1):
        params_dict = sim_info['params']
        sim_name = sim_info['name']
        
        print(f"\n{'='*80}")
        print(f"Simulation {idx}/{total_sims}: {sim_name}")
        print(f"   Varying: {sim_info['varied_param']} = {params_dict[sim_info['varied_param']]} ({sim_info['value_name']})")
        print(f"{'='*80}")
        
        # Add varied_param to params_dict for use in run_single_simulation
        params_dict['varied_param'] = sim_info['varied_param']
        
        result = run_single_simulation(params_dict, mesh_file, results_dir)
        result.update(params_dict)
        result['simulation_name'] = sim_name
        result['varied_param'] = sim_info['varied_param']
        result['value_name'] = sim_info['value_name']
        results.append(result)
        
        # Save intermediate results
        if idx % 5 == 0 or idx == total_sims:
            save_summary(results, summary_file)
    
    # Final summary
    save_summary(results, summary_file)
    
    # Print statistics
    successful = sum(1 for r in results if r['success'])
    failed = total_sims - successful
    
    print("\n" + "="*80)
    print(" PARAMETER SWEEP COMPLETE")
    print("="*80)
    print(f"✅ Successful: {successful}/{total_sims}")
    print(f"❌ Failed: {failed}/{total_sims}")
    print(f"📄 Summary saved to: {summary_file}")
    print(f"📁 Results in: {results_dir}")


def save_summary(results, summary_file):
    """Save results summary to CSV"""
    if not results:
        return
    
    # Get all parameter names
    param_names = list(NORMAL_PARAMS.keys())
    fieldnames = ['simulation_name', 'varied_param', 'value_name', 'filename', 'success'] + param_names + ['norm_Az', 'B_rms', 'error']
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)


if __name__ == "__main__":
    import numpy as np
    main()

