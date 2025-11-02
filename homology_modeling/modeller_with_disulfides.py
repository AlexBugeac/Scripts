#!/usr/bin/env python3
"""
MODELLER script for homology modeling with enforced disulfide bonds.
This script creates models with proper disulfide bond constraints.
"""

from modeller import *
from modeller.automodel import *
import os

class MyModel(AutoModel):
    def special_restraints(self, aln):
        """Add disulfide bond restraints based on template structures."""
        # E2 protein expected disulfide bonds (based on template analysis)
        # These are the cysteines that should form disulfide bonds
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
        
        # Add disulfide bond restraints
        for cys1, cys2 in disulfide_pairs:
            # Add distance restraint for SG-SG atoms
            self.restraints.add(
                Forms.Gaussian(group=Physical.xy_distance,
                              feature=Features.Distance(
                                  Atom(self.residues[f'{cys1}:'], 'SG'),
                                  Atom(self.residues[f'{cys2}:'], 'SG')),
                              mean=2.05, stdev=0.1))
            
            # Add angle restraints for proper disulfide geometry
            # CB-SG-SG angle should be around 104 degrees
            self.restraints.add(
                Forms.Gaussian(group=Physical.angle,
                              feature=Features.Angle(
                                  Atom(self.residues[f'{cys1}:'], 'CB'),
                                  Atom(self.residues[f'{cys1}:'], 'SG'),
                                  Atom(self.residues[f'{cys2}:'], 'SG')),
                              mean=1.815, stdev=0.087))  # 104 degrees in radians
            
            self.restraints.add(
                Forms.Gaussian(group=Physical.angle,
                              feature=Features.Angle(
                                  Atom(self.residues[f'{cys2}:'], 'CB'),
                                  Atom(self.residues[f'{cys2}:'], 'SG'),
                                  Atom(self.residues[f'{cys1}:'], 'SG')),
                              mean=1.815, stdev=0.087))  # 104 degrees in radians
            
            # Add dihedral restraint for SG-SG-CB-CA
            self.restraints.add(
                Forms.Gaussian(group=Physical.dihedral,
                              feature=Features.Dihedral(
                                  Atom(self.residues[f'{cys1}:'], 'SG'),
                                  Atom(self.residues[f'{cys2}:'], 'SG'),
                                  Atom(self.residues[f'{cys2}:'], 'CB'),
                                  Atom(self.residues[f'{cys2}:'], 'CA')),
                              mean=1.745, stdev=0.524))  # 100 degrees in radians

def run_modeling():
    """Run MODELLER with disulfide bond constraints."""
    
    # Set up environment
    env = Environ()
    env.io.atom_files_directory = ['.', '/home/alexb/Desktop/final/Modele/template_dual_clean/']
    
    # Create automodel class
    a = MyModel(env,
                alnfile='E2_dual_clean_final.pir',    # alignment file
                knowns=['8RJJ_B_clean', '8RK0_B_clean'],  # template codes
                sequence='E2_target',                   # target sequence code
                assess_methods=(assess.DOPE, assess.GA341))
    
    # Set modeling parameters
    a.starting_model = 1
    a.ending_model = 5
    a.md_level = refine.slow  # Use slow refinement for better disulfide geometry
    
    # Generate models
    print("Starting MODELLER with disulfide bond constraints...")
    a.make()
    
    # Assess models and find best one
    print("\nModel assessment:")
    best_model = None
    best_score = float('inf')
    
    for i in range(1, 6):
        model_file = f'E2_target.B{i:05d}0001.pdb'
        if os.path.exists(model_file):
            # Get DOPE score
            dope_score = None
            ga341_score = None
            
            # Read the model assessment
            try:
                with open(f'E2_target.D{i:05d}0001', 'r') as f:
                    for line in f:
                        if 'DOPE score' in line:
                            dope_score = float(line.split()[-1])
                        elif 'GA341 score' in line:
                            ga341_score = float(line.split()[-1])
            except:
                pass
                
            print(f"Model {i}: DOPE={dope_score}, GA341={ga341_score}")
            
            if dope_score and dope_score < best_score:
                best_score = dope_score
                best_model = model_file
    
    if best_model:
        # Copy best model to a more convenient name
        import shutil
        shutil.copy(best_model, 'E2_disulfide_model_best.pdb')
        print(f"\nBest model: {best_model} (DOPE: {best_score})")
        print("Copied to: E2_disulfide_model_best.pdb")
        
        # Add SSBOND records to the best model
        add_ssbond_records('E2_disulfide_model_best.pdb')
    
    return best_model

def add_ssbond_records(pdb_file):
    """Add SSBOND records to the PDB file for proper disulfide bond annotation."""
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
    
    # Read the PDB file
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    # Find where to insert SSBOND records (after HEADER, before ATOM)
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.startswith('ATOM'):
            insert_pos = i
            break
    
    # Create SSBOND records
    ssbond_lines = []
    for i, (cys1, cys2) in enumerate(disulfide_pairs, 1):
        ssbond_line = f"SSBOND {i:3d} CYS A {cys1:4d}    CYS A {cys2:4d}                          1555   1555  2.05\n"
        ssbond_lines.append(ssbond_line)
    
    # Insert SSBOND records
    lines = lines[:insert_pos] + ssbond_lines + lines[insert_pos:]
    
    # Write back to file
    with open(pdb_file, 'w') as f:
        f.writelines(lines)
    
    print(f"Added {len(disulfide_pairs)} SSBOND records to {pdb_file}")

if __name__ == "__main__":
    # Check if alignment file exists
    if not os.path.exists('E2_dual_clean_final.pir'):
        print("Error: E2_dual_clean_final.pir not found!")
        print("Please ensure the PIR alignment file is in the current directory.")
        exit(1)
    
    # Check if template files exist
    template_dir = '/home/alexb/Desktop/final/Modele/template_dual_clean/'
    for template in ['8RJJ_B_clean.pdb', '8RK0_B_clean.pdb']:
        template_path = os.path.join(template_dir, template)
        if not os.path.exists(template_path):
            print(f"Error: Template file {template} not found in {template_dir}")
            exit(1)
    
    print("Starting MODELLER with disulfide bond constraints...")
    print("This will enforce proper disulfide bond geometry during modeling.")
    print("Templates: 8RJJ_B_clean.pdb, 8RK0_B_clean.pdb")
    print("Expected disulfide bonds: 10 pairs")
    
    try:
        best_model = run_modeling()
        if best_model:
            print(f"\nModeling completed successfully!")
            print(f"Best model saved as: E2_disulfide_model_best.pdb")
            print("\nTo verify disulfide bonds, run:")
            print("python analyze_cys_distances.py E2_disulfide_model_best.pdb")
        else:
            print("Modeling failed - no models generated")
    except Exception as e:
        print(f"Error during modeling: {e}")
        print("Make sure MODELLER is properly installed and configured.")