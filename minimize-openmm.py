#!/usr/bin/env python3
"""
OpenMM minimization script with command-line arguments
"""

import argparse
import sys
from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Minimize molecular structures using OpenMM')
    
    parser.add_argument('input_file', 
                       help='Input structure file (PDB, CIF, etc.)')
    parser.add_argument('-o', '--output', 
                       default='minimized.pdb',
                       help='Output minimized structure file (default: minimized.pdb)')
    parser.add_argument('-f', '--forcefield', 
                       default='amber14-all.xml',
                       help='Force field XML file (default: amber14-all.xml)')
    parser.add_argument('-w', '--water', 
                       default='amber14/tip3pfb.xml',
                       help='Water model XML file (default: amber14/tip3pfb.xml)')
    parser.add_argument('--platform', 
                       default='CPU',
                       choices=['CPU', 'CUDA', 'OpenCL'],
                       help='Platform to use for simulation (default: CPU)')
    parser.add_argument('--tolerance', 
                       type=float, 
                       default=1.0,
                       help='Energy tolerance for minimization in kJ/mol (default: 1.0)')
    parser.add_argument('--max-iterations', 
                       type=int, 
                       default=1000,
                       help='Maximum number of minimization steps (default: 1000)')
    parser.add_argument('--implicit-solvent', 
                       action='store_true',
                       help='Use implicit solvent (GBn2) instead of explicit water')
    parser.add_argument('--add-hydrogens', 
                       action='store_true',
                       help='Add missing hydrogen atoms')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()

def setup_system(pdb, forcefield_files, implicit_solvent=False, add_hydrogens=False):
    """Set up the molecular system"""
    # Load force field
    forcefield = ForceField(*forcefield_files)
    
    # Add hydrogens if requested
    if add_hydrogens:
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)
        topology = modeller.topology
        positions = modeller.positions
    else:
        topology = pdb.topology
        positions = pdb.positions
    
    # Create system
    if implicit_solvent:
        system = forcefield.createSystem(topology, 
                                       nonbondedMethod=NoCutoff)
    else:
        system = forcefield.createSystem(topology, 
                                       nonbondedMethod=PME,
                                       nonbondedCutoff=1*nanometer,
                                       constraints=HBonds)
    
    return system, topology, positions

def minimize_structure(system, topology, positions, platform_name, tolerance, max_iterations, verbose=False):
    """Perform energy minimization"""
    # Set up platform
    platform = Platform.getPlatformByName(platform_name)
    
    # Create integrator (not used for minimization but required)
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    
    # Create simulation
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    
    # Get initial energy
    initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    
    if verbose:
        print(f"Initial potential energy: {initial_energy}")
        print(f"Starting minimization with tolerance {tolerance} kJ/mol...")
    
    # Minimize
    simulation.minimizeEnergy(tolerance=tolerance, 
                        maxIterations=max_iterations)
    
    # Get final energy and positions
    final_state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = final_state.getPotentialEnergy()
    final_positions = final_state.getPositions()
    
    if verbose:
        print(f"Final potential energy: {final_energy}")
        print(f"Energy change: {final_energy - initial_energy}")
    
    return final_positions, initial_energy, final_energy

def main():
    """Main function"""
    args = parse_arguments()
    
    if args.verbose:
        print(f"Loading structure from: {args.input_file}")
    
    try:
        # Load input structure
        pdb = PDBFile(args.input_file)
        
        # Set up force field files
        if args.implicit_solvent:
            forcefield_files = [args.forcefield, 'implicit/gbn2.xml']
        else:
            forcefield_files = [args.forcefield, args.water]
        
        # Set up system
        system, topology, positions = setup_system(pdb, forcefield_files, 
                                                   args.implicit_solvent, 
                                                   args.add_hydrogens)
        
        # Minimize
        final_positions, initial_energy, final_energy = minimize_structure(
            system, topology, positions, args.platform, 
            args.tolerance, args.max_iterations, args.verbose)
        
        # Save minimized structure
        with open(args.output, 'w') as f:
            PDBFile.writeFile(topology, final_positions, f)
        
        print(f"Minimization completed!")
        print(f"Initial energy: {initial_energy}")
        print(f"Final energy: {final_energy}")
        print(f"Energy change: {final_energy - initial_energy}")
        print(f"Minimized structure saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()