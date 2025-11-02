#!/usr/bin/env python3
"""
Quick test MD simulation using robosample - 1 minute runtime
Modified from simulate.equilibrium.py for rapid testing
"""

import os
import sys
import numpy as np
from robosample import Context, Sampler, Integrator
import flexor3

def quick_test_simulation(prmtop_file, rst7_file, output_prefix="quick_test"):
    """Run a very short MD simulation for testing"""
    
    print("🧪 Starting quick test simulation...")
    print(f"Topology: {prmtop_file}")
    print(f"Coordinates: {rst7_file}")
    
    try:
        # Create context with minimal settings for quick test
        ctx = Context(prmtop_file, rst7_file)
        
        # Set up very short simulation parameters
        temperature = 300.0  # K
        timestep = 2.0       # fs
        
        # Very short steps for 1-minute test
        minimize_steps = 100      # ~10 seconds
        heat_steps = 250         # ~20 seconds  
        equil_steps = 250        # ~20 seconds
        prod_steps = 250         # ~20 seconds
        
        print(f"⚙️  Simulation settings:")
        print(f"   Temperature: {temperature} K")
        print(f"   Timestep: {timestep} fs")
        print(f"   Minimize: {minimize_steps} steps")
        print(f"   Heating: {heat_steps} steps")
        print(f"   Equilibration: {equil_steps} steps")
        print(f"   Production: {prod_steps} steps")
        
        # Minimization
        print("🔧 Energy minimization...")
        ctx.minimize(maxIterations=minimize_steps)
        
        # Heating phase
        print("🌡️  Heating to target temperature...")
        ctx.setTemperature(temperature)
        integrator = Integrator('langevin', timestep=timestep, temperature=temperature)
        ctx.setIntegrator(integrator)
        
        # Quick heating
        for step in range(heat_steps):
            ctx.step()
            if step % 50 == 0:
                temp = ctx.getTemperature()
                energy = ctx.getPotentialEnergy()
                print(f"   Step {step}: T={temp:.1f}K, E={energy:.1f} kJ/mol")
        
        # Equilibration
        print("⚖️  Equilibration...")
        for step in range(equil_steps):
            ctx.step()
            if step % 50 == 0:
                temp = ctx.getTemperature()
                energy = ctx.getPotentialEnergy()
                print(f"   Step {step}: T={temp:.1f}K, E={energy:.1f} kJ/mol")
        
        # Production run with enhanced sampling
        print("🎯 Production run with enhanced sampling...")
        
        # Set up simple enhanced sampling (just one world for quick test)
        sampler = Sampler()
        sampler.addWorld("cartesian", weight=1.0)
        
        # Production steps
        for step in range(prod_steps):
            sampler.step(ctx)
            
            if step % 50 == 0:
                temp = ctx.getTemperature()
                energy = ctx.getPotentialEnergy()
                print(f"   Step {step}: T={temp:.1f}K, E={energy:.1f} kJ/mol")
        
        # Save final structure
        final_pdb = f"{output_prefix}_final.pdb"
        ctx.saveState(final_pdb)
        print(f"💾 Final structure saved: {final_pdb}")
        
        print("✅ Quick test simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Check for input files
    prmtop_file = "E2_test.prmtop"
    rst7_file = "E2_test.rst7"
    
    if not os.path.exists(prmtop_file):
        print(f"❌ Topology file not found: {prmtop_file}")
        print("Run quick_test_prep.py first to prepare files.")
        return
    
    if not os.path.exists(rst7_file):
        print(f"❌ Coordinate file not found: {rst7_file}")
        print("Run quick_test_prep.py first to prepare files.")
        return
    
    # Run the simulation
    success = quick_test_simulation(prmtop_file, rst7_file)
    
    if success:
        print("🎉 Test completed! Check quick_test_final.pdb for results.")
    else:
        print("❌ Test failed. Check error messages above.")

if __name__ == "__main__":
    main()