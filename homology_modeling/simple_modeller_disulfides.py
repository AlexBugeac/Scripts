#!/usr/bin/env python3
"""
Simple MODELLER script for homology modeling with disulfide bond post-processing.
This creates models and then adds proper disulfide bond annotations.
"""

import os
import sys

# Try to import MODELLER
try:
    from modeller import *
    from modeller.automodel import *
except ImportError:
    print("Error: MODELLER not found. Please install MODELLER first.")
    print("conda install -c salilab modeller")
    sys.exit(1)

def run_simple_modeling():
    """Run basic MODELLER and then add disulfide bonds."""
    
    # Set up environment
    env = Environ()
    env.io.atom_files_directory = ['.', '/home/alexb/Desktop/final/Templates/']
    
    # Create automodel
    a = AutoModel(env,
                  alnfile='E2_dual_clean_final.pir',
                  knowns=['8RJJ_B_CLEAN', '8RK0_B_CLEAN'],
                  sequence='E2_target',
                  assess_methods=(assess.DOPE, assess.GA341))
    
    # Model parameters
    a.starting_model = 1
    a.ending_model = 3  # Generate 3 models for faster testing
    a.md_level = refine.slow  # Use slow refinement
    
    print("Running MODELLER...")
    a.make()
    
    # Find best model
    best_model = None
    best_score = float('inf')
    
    print("\nModel assessment:")
    for i in range(1, 4):
        model_file = f'E2_target.B{i:05d}0001.pdb'
        if os.path.exists(model_file):
            # Try to get DOPE score from log
            dope_score = None
            try:
                # Look for assessment file
                assess_file = f'E2_target.D{i:05d}0001'
                if os.path.exists(assess_file):
                    with open(assess_file, 'r') as f:
                        content = f.read()
                        # Extract DOPE score
                        for line in content.split('\n'):
                            if 'DOPE score' in line:
                                dope_score = float(line.split()[-1])
                                break
            except:
                dope_score = None
            
            print(f"Model {i}: {model_file} (DOPE: {dope_score})")
            
            if dope_score and dope_score < best_score:
                best_score = dope_score
                best_model = model_file
    
    if best_model:
        # Copy and process best model
        import shutil
        output_file = 'E2_model_with_disulfides.pdb'
        shutil.copy(best_model, output_file)
        
        # Add disulfide bond annotations
        add_disulfide_annotations(output_file)
        
        print(f"\nBest model: {best_model}")
        print(f"Output with disulfides: {output_file}")
        return output_file
    else:
        print("No models generated successfully.")
        return None

def add_disulfide_annotations(pdb_file):
    """Add SSBOND and CONECT records for disulfide bonds."""
    
    # Expected disulfide pairs for E2 protein
    disulfide_pairs = [
        (46, 120),   # Cys46-Cys120
        (69, 76),    # Cys69-Cys76  
        (103, 111),  # Cys103-Cys111
        (125, 169),  # Cys125-Cys169
        (181, 186),  # Cys181-Cys186
        (198, 202),  # Cys198-Cys202
        (214, 224),  # Cys214-Cys224
        (237, 261),  # Cys237-Cys261
        (269, 294),  # Cys269-Cys294
        (349, 351),  # Cys349-Cys351
    ]
    
    # Read PDB file
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    # Find cysteine SG atoms and their line numbers
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
    
    # Find insertion point for SSBOND records (after header, before atoms)
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.startswith('ATOM'):
            insert_pos = i
            break
    
    # Create SSBOND records
    ssbond_lines = []
    conect_lines = []
    
    bonds_added = 0
    for bond_num, (cys1, cys2) in enumerate(disulfide_pairs, 1):
        if cys1 in cys_atoms and cys2 in cys_atoms:
            # Add SSBOND record
            ssbond_line = f"SSBOND {bond_num:3d} CYS A {cys1:4d}    CYS A {cys2:4d}                          1555   1555  2.05\n"
            ssbond_lines.append(ssbond_line)
            
            # Add CONECT records for visualization
            serial1 = cys_atoms[cys1]['serial']
            serial2 = cys_atoms[cys2]['serial']
            conect_lines.append(f"CONECT{serial1:5d}{serial2:5d}\n")
            conect_lines.append(f"CONECT{serial2:5d}{serial1:5d}\n")
            
            bonds_added += 1
    
    # Insert SSBOND records after header
    lines = lines[:insert_pos] + ssbond_lines + lines[insert_pos:]
    
    # Add CONECT records at the end (before END if present)
    if lines and lines[-1].startswith('END'):
        lines = lines[:-1] + conect_lines + [lines[-1]]
    else:
        lines.extend(conect_lines)
    
    # Write back to file
    with open(pdb_file, 'w') as f:
        f.writelines(lines)
    
    print(f"Added {bonds_added} disulfide bonds to {pdb_file}")
    return bonds_added

def main():
    """Main function."""
    
    # Check prerequisites
    if not os.path.exists('E2_dual_clean_final.pir'):
        print("Error: E2_dual_clean_final.pir not found!")
        print("Please ensure the PIR alignment file is in the current directory.")
        return
    
    template_dir = '/home/alexb/Desktop/final/Templates/'
    for template in ['8RJJ_B_clean.pdb', '8RK0_B_clean.pdb']:
        template_path = os.path.join(template_dir, template)
        if not os.path.exists(template_path):
            print(f"Error: Template {template} not found in {template_dir}")
            return
    
    print("Starting MODELLER homology modeling...")
    print("Templates: 8RJJ_B_clean, 8RK0_B_clean")
    print("Target: E2_target")
    
    try:
        output_file = run_simple_modeling()
        if output_file:
            print(f"\nModeling completed successfully!")
            print(f"Output file: {output_file}")
            print("\nTo analyze disulfide bonds:")
            print(f"python analyze_cys_distances.py {output_file}")
            print("\nTo view in PyMOL:")
            print(f"pymol {output_file}")
    except Exception as e:
        print(f"Error during modeling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()