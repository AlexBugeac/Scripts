#!/usr/bin/env python3
"""
Force add disulfide bonds to PDB file regardless of distance.
"""

import sys

def force_add_disulfides(input_pdb, output_pdb):
    """Force add disulfide bonds for E2 protein."""
    
    # Expected disulfide pairs for E2
    disulfide_pairs = [
        (46, 69),   # Far but functional
        (76, 103),  # ~12 Å
        (111, 125), # ~15 Å  
        (120, 181), # ~13 Å
        (169, 186), # ~9.6 Å - closest
        (198, 214), # ~17 Å
        (202, 224), # ~17 Å
        (237, 269), # ~18 Å
        (261, 294), # ~25 Å
        (349, 351), # ~8.7 Å - second closest
    ]
    
    # Read PDB file
    with open(input_pdb, 'r') as f:
        lines = f.readlines()
    
    # Find cysteine SG atoms
    cys_atoms = {}
    for i, line in enumerate(lines):
        if line.startswith('ATOM') and line[12:16].strip() == 'SG':
            try:
                resnum = int(line[22:26].strip())
                resname = line[17:20].strip()
                if resname == 'CYS':
                    atom_serial = int(line[6:11].strip())
                    cys_atoms[resnum] = {'line': i, 'serial': atom_serial}
            except:
                continue
    
    # Find insertion point for SSBOND records
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.startswith('ATOM'):
            insert_pos = i
            break
    
    # Create SSBOND and CONECT records
    ssbond_lines = []
    conect_lines = []
    
    bonds_added = 0
    for bond_num, (cys1, cys2) in enumerate(disulfide_pairs, 1):
        if cys1 in cys_atoms and cys2 in cys_atoms:
            # Add SSBOND record
            ssbond_line = f"SSBOND {bond_num:3d} CYS A {cys1:4d}    CYS A {cys2:4d}                          1555   1555  2.05\n"
            ssbond_lines.append(ssbond_line)
            
            # Add CONECT records
            serial1 = cys_atoms[cys1]['serial']
            serial2 = cys_atoms[cys2]['serial']
            conect_lines.append(f"CONECT{serial1:5d}{serial2:5d}\n")
            conect_lines.append(f"CONECT{serial2:5d}{serial1:5d}\n")
            
            bonds_added += 1
            print(f"Added bond {bond_num}: CYS {cys1} - CYS {cys2}")
    
    # Insert SSBOND records
    lines = lines[:insert_pos] + ssbond_lines + lines[insert_pos:]
    
    # Add CONECT records at the end
    if lines and lines[-1].startswith('END'):
        lines = lines[:-1] + conect_lines + [lines[-1]]
    else:
        lines.extend(conect_lines)
    
    # Write output
    with open(output_pdb, 'w') as f:
        f.writelines(lines)
    
    print(f"\nForce added {bonds_added} disulfide bonds to {output_pdb}")
    print("These bonds may have large distances and require structural refinement.")
    return bonds_added

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python force_add_disulfides.py input.pdb output.pdb")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        bonds_added = force_add_disulfides(input_file, output_file)
        print(f"\nSuccess! File saved as {output_file}")
        print("View in PyMOL to see disulfide bond annotations.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)