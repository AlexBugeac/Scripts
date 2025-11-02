#!/usr/bin/env python3
"""
Quick test script to prepare E2-n.pdb for a short MD simulation
"""

import os
import subprocess
import tempfile

def prepare_amber_files(pdb_file, output_name="E2_test"):
    """Prepare Amber topology and coordinate files from PDB"""
    
    # Create tleap input script
    tleap_script = f"""
source leaprc.protein.ff14SB
source leaprc.water.tip3p

# Load the PDB file
mol = loadpdb {pdb_file}

# Add hydrogen atoms and solve any missing atoms
addhydrogens mol

# Create a water box around the protein (minimal for quick test)
solvatebox mol TIP3PBOX 8.0

# Add ions to neutralize the system
addionsrand mol Na+ 0
addionsrand mol Cl- 0

# Save topology and coordinates
saveamberparm mol {output_name}.prmtop {output_name}.rst7

# Save PDB for visualization
savepdb mol {output_name}_solvated.pdb

quit
"""
    
    # Write tleap script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
        f.write(tleap_script)
        tleap_input = f.name
    
    try:
        # Run tleap
        print(f"Running tleap to prepare {pdb_file}...")
        result = subprocess.run(['tleap', '-f', tleap_input], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Amber files prepared successfully!")
            print(f"Created: {output_name}.prmtop, {output_name}.rst7")
            return f"{output_name}.prmtop", f"{output_name}.rst7"
        else:
            print("❌ Error running tleap:")
            print(result.stderr)
            return None, None
            
    finally:
        # Clean up temporary file
        os.unlink(tleap_input)

if __name__ == "__main__":
    pdb_file = "/home/alexb/Desktop/final/Modele/E2-n.pdb"
    
    if os.path.exists(pdb_file):
        print(f"Preparing {pdb_file} for MD simulation...")
        prmtop, rst7 = prepare_amber_files(pdb_file)
        
        if prmtop and rst7:
            print(f"Ready for simulation with:")
            print(f"  Topology: {prmtop}")
            print(f"  Coordinates: {rst7}")
    else:
        print(f"❌ PDB file not found: {pdb_file}")