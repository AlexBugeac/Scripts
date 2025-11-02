#!/usr/bin/env python3
"""
Structural Quality Analyzer
Analyzes actual structural integrity of models beyond scoring metrics
"""

import os
import sys
import argparse
from pathlib import Path

def analyze_pdb_structure(pdb_file):
    """Analyze structural integrity of a PDB file"""
    print(f"\n🔍 Analyzing: {pdb_file}")
    
    if not os.path.exists(pdb_file):
        print(f"❌ File not found: {pdb_file}")
        return
    
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    # Extract ATOM records
    atoms = [line for line in lines if line.startswith('ATOM')]
    residues = {}
    
    for line in atoms:
        res_num = int(line[22:26].strip())
        res_name = line[17:20].strip()
        atom_name = line[12:16].strip()
        
        if res_num not in residues:
            residues[res_num] = {'name': res_name, 'atoms': [], 'backbone': []}
        
        residues[res_num]['atoms'].append(atom_name)
        if atom_name in ['N', 'CA', 'C', 'O']:
            residues[res_num]['backbone'].append(atom_name)
    
    print(f"📊 Total residues: {len(residues)}")
    print(f"📊 Total atoms: {len(atoms)}")
    
    # Check for missing backbone atoms
    missing_backbone = []
    incomplete_residues = []
    
    for res_num, res_data in residues.items():
        backbone_atoms = set(res_data['backbone'])
        expected_backbone = {'N', 'CA', 'C', 'O'}
        
        if not expected_backbone.issubset(backbone_atoms):
            missing = expected_backbone - backbone_atoms
            incomplete_residues.append((res_num, res_data['name'], missing))
    
    if incomplete_residues:
        print(f"⚠️  {len(incomplete_residues)} residues with missing backbone atoms:")
        for res_num, res_name, missing in incomplete_residues[:10]:  # Show first 10
            print(f"   Residue {res_num} ({res_name}): missing {missing}")
        if len(incomplete_residues) > 10:
            print(f"   ... and {len(incomplete_residues) - 10} more")
    else:
        print("✅ All residues have complete backbone")
    
    # Check for large gaps in residue numbering
    res_numbers = sorted(residues.keys())
    gaps = []
    
    for i in range(len(res_numbers) - 1):
        gap = res_numbers[i+1] - res_numbers[i]
        if gap > 1:
            gaps.append((res_numbers[i], res_numbers[i+1], gap-1))
    
    if gaps:
        print(f"⚠️  {len(gaps)} gaps in structure:")
        for start, end, gap_size in gaps:
            print(f"   Gap between residues {start}-{end} ({gap_size} missing residues)")
    else:
        print("✅ No gaps in residue numbering")
    
    # Check coordinate quality (very basic)
    coords = []
    for line in atoms:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coords.append((x, y, z))
    
    # Look for obviously problematic coordinates
    extreme_coords = [coord for coord in coords if any(abs(c) > 500 for c in coord)]
    
    if extreme_coords:
        print(f"⚠️  {len(extreme_coords)} atoms with extreme coordinates (>500Å)")
    else:
        print("✅ Coordinates appear reasonable")
    
    return {
        'total_residues': len(residues),
        'total_atoms': len(atoms),
        'incomplete_residues': len(incomplete_residues),
        'gaps': len(gaps),
        'extreme_coords': len(extreme_coords)
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze structural quality of models")
    parser.add_argument("models", nargs="+", help="PDB files to analyze")
    
    args = parser.parse_args()
    
    print("🧬 Structural Quality Analysis")
    print("=" * 50)
    
    results = {}
    for model_file in args.models:
        results[model_file] = analyze_pdb_structure(model_file)
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    for model_file, result in results.items():
        model_name = os.path.basename(model_file)
        print(f"\n{model_name}:")
        print(f"  Residues: {result['total_residues']}")
        print(f"  Atoms: {result['total_atoms']}")
        
        if result['incomplete_residues'] > 0:
            print(f"  ❌ {result['incomplete_residues']} incomplete residues")
        else:
            print(f"  ✅ Complete backbone")
            
        if result['gaps'] > 0:
            print(f"  ❌ {result['gaps']} structural gaps")
        else:
            print(f"  ✅ No gaps")
            
        if result['extreme_coords'] > 0:
            print(f"  ❌ {result['extreme_coords']} problematic coordinates")
        else:
            print(f"  ✅ Good coordinates")

if __name__ == "__main__":
    main()