#!/usr/bin/env python3
"""
Add disulfide bonds to homology models based on template patterns
"""

import sys
import re
from pathlib import Path

def calculate_distance(atom1_coords, atom2_coords):
    """Calculate distance between two atoms"""
    x1, y1, z1 = atom1_coords
    x2, y2, z2 = atom2_coords
    return ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5

def parse_atom_line(line):
    """Parse PDB ATOM line to extract coordinates and info"""
    atom_name = line[12:16].strip()
    res_name = line[17:20].strip()
    chain_id = line[21].strip()
    res_num = int(line[22:26].strip())
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    atom_num = int(line[6:11].strip())
    
    return {
        'atom_num': atom_num,
        'atom_name': atom_name,
        'res_name': res_name,
        'chain_id': chain_id,
        'res_num': res_num,
        'coords': (x, y, z),
        'line': line
    }

def add_disulfide_bonds(input_pdb, output_pdb, disulfide_pairs=None):
    """
    Add disulfide bonds to a PDB file
    
    Args:
        input_pdb (str): Input PDB file path
        output_pdb (str): Output PDB file path  
        disulfide_pairs (list): List of tuples (res1, res2) for disulfide bonds
    """
    
    # Default E2 disulfide bonds based on HCV E2 structure
    if disulfide_pairs is None:
        # These are the typical HCV E2 disulfide bonds (adjusted for model numbering)
        disulfide_pairs = [
            (46, 69),   # First disulfide bond
            (76, 103),  # Second disulfide bond  
            (111, 125), # Third disulfide bond
            (120, 181), # Fourth disulfide bond
            (169, 186), # Fifth disulfide bond
            (198, 214), # Sixth disulfide bond
            (202, 224), # Seventh disulfide bond
            (237, 269), # Eighth disulfide bond
            (261, 294), # Additional bonds
            (349, 351)  # Terminal disulfide
        ]
    
    print(f"Adding disulfide bonds to {input_pdb}")
    print(f"Expected disulfide pairs: {disulfide_pairs}")
    
    # Read PDB file and find cysteine SG atoms
    atoms = []
    cys_sg_atoms = {}
    
    with open(input_pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_info = parse_atom_line(line)
                atoms.append(atom_info)
                
                # Store cysteine SG atoms
                if atom_info['res_name'] == 'CYS' and atom_info['atom_name'] == 'SG':
                    cys_sg_atoms[atom_info['res_num']] = atom_info
    
    print(f"Found {len(cys_sg_atoms)} cysteine SG atoms at positions: {sorted(cys_sg_atoms.keys())}")
    
    # Verify disulfide pairs and calculate distances
    valid_disulfides = []
    conect_records = []
    
    for i, (res1, res2) in enumerate(disulfide_pairs, 1):
        if res1 in cys_sg_atoms and res2 in cys_sg_atoms:
            atom1 = cys_sg_atoms[res1]
            atom2 = cys_sg_atoms[res2]
            distance = calculate_distance(atom1['coords'], atom2['coords'])
            
            print(f"Disulfide {i}: CYS {res1} - CYS {res2}, distance: {distance:.2f} Å")
            
            if distance < 3.0:  # Reasonable disulfide bond distance
                valid_disulfides.append((i, res1, res2, distance))
                # Create CONECT records
                conect_records.append(f"CONECT{atom1['atom_num']:5d}{atom2['atom_num']:5d}")
                conect_records.append(f"CONECT{atom2['atom_num']:5d}{atom1['atom_num']:5d}")
            else:
                print(f"  WARNING: Distance {distance:.2f} Å is too large for disulfide bond")
        else:
            missing = []
            if res1 not in cys_sg_atoms:
                missing.append(str(res1))
            if res2 not in cys_sg_atoms:
                missing.append(str(res2))
            print(f"  WARNING: Missing cysteine(s) at position(s): {', '.join(missing)}")
    
    # Write output PDB with disulfide bonds
    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        header_written = False
        atoms_written = False
        
        for line in infile:
            if line.startswith('ATOM') and not header_written:
                # Write SSBOND records before atoms
                for i, res1, res2, distance in valid_disulfides:
                    ssbond_line = f"SSBOND{i:4d} CYS A{res1:4d}    CYS A{res2:4d}             1555   1555  {distance:.2f}\n"
                    outfile.write(ssbond_line)
                header_written = True
            
            # Write the original line
            outfile.write(line)
            
            # Add CONECT records after all ATOM records
            if line.startswith('TER') or (line.startswith('END') and not atoms_written):
                for conect_line in conect_records:
                    outfile.write(conect_line + '\n')
                atoms_written = True
    
    print(f"\nOutput written to: {output_pdb}")
    print(f"Added {len(valid_disulfides)} disulfide bonds")
    print(f"Added {len(conect_records)} CONECT records")
    
    return len(valid_disulfides)

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 3:
        print("Usage: python add_disulfide_bonds.py <input.pdb> <output.pdb> [custom_pairs]")
        print("Example: python add_disulfide_bonds.py model.pdb model_with_ss.pdb")
        print("Custom pairs: python add_disulfide_bonds.py model.pdb model_with_ss.pdb '46,69;76,103'")
        sys.exit(1)
    
    input_pdb = sys.argv[1]
    output_pdb = sys.argv[2]
    
    # Parse custom disulfide pairs if provided
    custom_pairs = None
    if len(sys.argv) > 3:
        pairs_str = sys.argv[3]
        custom_pairs = []
        for pair in pairs_str.split(';'):
            res1, res2 = map(int, pair.split(','))
            custom_pairs.append((res1, res2))
        print(f"Using custom disulfide pairs: {custom_pairs}")
    
    if not Path(input_pdb).exists():
        print(f"Error: Input file {input_pdb} not found!")
        sys.exit(1)
    
    num_bonds = add_disulfide_bonds(input_pdb, output_pdb, custom_pairs)
    print(f"\nSuccess! Added {num_bonds} disulfide bonds to {output_pdb}")

if __name__ == "__main__":
    main()