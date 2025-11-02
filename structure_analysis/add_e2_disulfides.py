#!/usr/bin/env python3
"""
Add correct E2 disulfide bonds to E2-n.pdb model based on template structure.
Uses the proper disulfide bond mapping from 8RJJ template.
"""

import sys
import os

def add_e2_disulfide_bonds(input_pdb, output_pdb):
    """Add E2 disulfide bonds based on template structure mapping."""
    
    # Correct E2 disulfide pairs from 8RJJ template (1-based numbering)
    # Template positions converted: (template_pos - 383) for 1-based
    disulfide_pairs = [
        (46, 121),   # CYS 429 - CYS 504 in template
        (69, 243),   # CYS 452 - CYS 626 in template  
        (76, 104),   # CYS 459 - CYS 487 in template
        (112, 182),  # CYS 495 - CYS 565 in template
        (126, 170),  # CYS 509 - CYS 553 in template
        (187, 220),  # CYS 570 - CYS 603 in template
        (204, 208),  # CYS 587 - CYS 591 in template
        (230, 267),  # CYS 613 - CYS 650 in template
    ]
    
    print(f"Adding E2 disulfide bonds to {input_pdb}")
    print("Template-based disulfide pairs:")
    for i, (cys1, cys2) in enumerate(disulfide_pairs, 1):
        print(f"  {i}. CYS {cys1} - CYS {cys2}")
    
    # Read PDB file
    with open(input_pdb, 'r') as f:
        lines = f.readlines()
    
    # Find cysteine SG atoms and their distances
    cys_atoms = {}
    for i, line in enumerate(lines):
        if line.startswith('ATOM') and line[12:16].strip() == 'SG':
            try:
                resnum = int(line[22:26].strip())
                resname = line[17:20].strip()
                if resname == 'CYS':
                    atom_serial = int(line[6:11].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    cys_atoms[resnum] = {
                        'line': i, 
                        'serial': atom_serial,
                        'coords': (x, y, z)
                    }
            except:
                continue
    
    print(f"\nFound {len(cys_atoms)} cysteine SG atoms in model:")
    print(f"Positions: {sorted(cys_atoms.keys())}")
    
    # Calculate distances for expected pairs
    import math
    def distance(pos1, pos2):
        return math.sqrt(sum((a-b)**2 for a, b in zip(pos1, pos2)))
    
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
    print(f"\nDisulfide bond analysis:")
    for bond_num, (cys1, cys2) in enumerate(disulfide_pairs, 1):
        if cys1 in cys_atoms and cys2 in cys_atoms:
            # Calculate distance
            coords1 = cys_atoms[cys1]['coords']
            coords2 = cys_atoms[cys2]['coords']
            dist = distance(coords1, coords2)
            
            # Add records regardless of distance (for visualization)
            ssbond_line = f"SSBOND {bond_num:3d} CYS A {cys1:4d}    CYS A {cys2:4d}                          1555   1555  2.05\n"
            ssbond_lines.append(ssbond_line)
            
            # Add CONECT records
            serial1 = cys_atoms[cys1]['serial']
            serial2 = cys_atoms[cys2]['serial']
            conect_lines.append(f"CONECT{serial1:5d}{serial2:5d}\n")
            conect_lines.append(f"CONECT{serial2:5d}{serial1:5d}\n")
            
            bonds_added += 1
            status = "✓ GOOD" if dist < 3.0 else "⚠ LARGE" if dist < 5.0 else "✗ TOO FAR"
            print(f"  {bond_num}. CYS {cys1:3d} - CYS {cys2:3d}: {dist:6.2f} Å  {status}")
        else:
            missing = []
            if cys1 not in cys_atoms:
                missing.append(str(cys1))
            if cys2 not in cys_atoms:
                missing.append(str(cys2))
            print(f"  {bond_num}. CYS {cys1:3d} - CYS {cys2:3d}: MISSING CYS {', '.join(missing)}")
    
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
    
    print(f"\nAdded {bonds_added} disulfide bonds to {output_pdb}")
    return bonds_added

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_e2_disulfides.py input.pdb output.pdb")
        print("Example: python add_e2_disulfides.py E2-n.pdb E2-n_with_disulfides.pdb")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        sys.exit(1)
    
    try:
        bonds_added = add_e2_disulfide_bonds(input_file, output_file)
        print(f"\nSuccess! Output saved as {output_file}")
        print("View in PyMOL to see disulfide bond annotations.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)