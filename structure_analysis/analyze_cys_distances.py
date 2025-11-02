#!/usr/bin/env python3
"""
Analyze cysteine distances in PDB to find potential disulfide bonds
"""

import sys
import math

def parse_atom_line(line):
    """Parse PDB ATOM line"""
    atom_name = line[12:16].strip()
    res_name = line[17:20].strip()
    res_num = int(line[22:26].strip())
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    
    return {
        'atom_name': atom_name,
        'res_name': res_name,
        'res_num': res_num,
        'coords': (x, y, z)
    }

def calculate_distance(coords1, coords2):
    """Calculate distance between two points"""
    x1, y1, z1 = coords1
    x2, y2, z2 = coords2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def analyze_cysteine_distances(pdb_file):
    """Analyze all cysteine SG-SG distances"""
    
    cys_sg_atoms = {}
    
    # Read PDB and collect cysteine SG atoms
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_info = parse_atom_line(line)
                if atom_info['res_name'] == 'CYS' and atom_info['atom_name'] == 'SG':
                    cys_sg_atoms[atom_info['res_num']] = atom_info
    
    print(f"Found {len(cys_sg_atoms)} cysteine SG atoms")
    print(f"Positions: {sorted(cys_sg_atoms.keys())}")
    print()
    
    # Calculate all pairwise distances
    distances = []
    residues = sorted(cys_sg_atoms.keys())
    
    for i, res1 in enumerate(residues):
        for res2 in residues[i+1:]:
            coords1 = cys_sg_atoms[res1]['coords']
            coords2 = cys_sg_atoms[res2]['coords']
            distance = calculate_distance(coords1, coords2)
            distances.append((res1, res2, distance))
    
    # Sort by distance
    distances.sort(key=lambda x: x[2])
    
    print("All Cysteine SG-SG distances (sorted by distance):")
    print("=" * 50)
    
    potential_bonds = []
    for res1, res2, distance in distances:
        status = ""
        if distance < 3.0:
            status = "  ✓ DISULFIDE BOND"
            potential_bonds.append((res1, res2))
        elif distance < 5.0:
            status = "  ? POSSIBLE (needs refinement)"
        
        print(f"CYS {res1:3d} - CYS {res2:3d}: {distance:6.2f} Å{status}")
    
    print()
    print(f"Found {len(potential_bonds)} probable disulfide bonds:")
    for res1, res2 in potential_bonds:
        print(f"  CYS {res1} - CYS {res2}")
    
    return potential_bonds

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_cys_distances.py <pdb_file>")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    potential_bonds = analyze_cysteine_distances(pdb_file)