#!/usr/bin/env python3

"""
MODELLER Script: Create E2 models with 41-46 disulfide bond
Two models: 1) Full relaxation, 2) Constrained (residues 51+ fixed)
"""

from modeller import *
from modeller.automodel import *

# Create the sequence from the PDB file
def extract_sequence_from_pdb(pdb_file):
    """Extract sequence from PDB file"""
    sequence = ""
    prev_resnum = -999
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                resnum = int(line[22:26])
                if resnum != prev_resnum:
                    resname = line[17:20].strip()
                    # Convert 3-letter to 1-letter code
                    aa_map = {
                        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                    }
                    sequence += aa_map.get(resname, 'X')
                    prev_resnum = resnum
    
    return sequence

class E2_Model_FullRelax(AutoModel):
    def special_patches(self, aln):
        """Add 41-46 disulfide bond - full structure relaxation"""
        try:
            self.patch(residue_type='DISU', 
                      residues=(self.residues['41:A'], 
                               self.residues['46:A']))
            print("Added disulfide bond: 41-46 (full relaxation)")
        except Exception as e:
            print(f"Failed to add disulfide bond 41-46: {e}")

class E2_Model_28_41(AutoModel):
    def special_patches(self, aln):
        """Add 28-41 disulfide bond"""
        try:
            self.patch(residue_type='DISU', 
                      residues=(self.residues['28:A'], 
                               self.residues['41:A']))
            print("Added disulfide bond: 28-41")
        except Exception as e:
            print(f"Failed to add disulfide bond 28-41: {e}")

def create_pir_files():
    """Create PIR alignment files"""
    
    # Extract sequence from PDB
    sequence = extract_sequence_from_pdb('/home/alexb/Desktop/final/Modele/E2-m424-429.pdb')
    
    # Create PIR for full relaxation
    with open('E2_41_46_full.pir', 'w') as f:
        f.write(">P1;E2-m424-429\n")
        f.write("structureX:E2-m424-429:1:A:363:A:E2 protein:HCV: 2.00: 0.19\n")
        # Split sequence into lines of 50 characters
        for i in range(0, len(sequence), 50):
            f.write(sequence[i:i+50] + "\n")
        f.write("*\n")
        f.write(">P1;E2_target_full\n")
        f.write("sequence:E2_target_full:1:A:363:A:E2 protein:HCV: 0.00: 0.00\n")
        for i in range(0, len(sequence), 50):
            f.write(sequence[i:i+50] + "\n")
        f.write("*\n")
    
    # Create PIR for 28-41 model  
    with open('E2_28_41.pir', 'w') as f:
        f.write(">P1;E2-m424-429\n")
        f.write("structureX:E2-m424-429:1:A:363:A:E2 protein:HCV: 2.00: 0.19\n")
        for i in range(0, len(sequence), 50):
            f.write(sequence[i:i+50] + "\n")
        f.write("*\n")
        f.write(">P1;E2_target_28_41\n")
        f.write("sequence:E2_target_28_41:1:A:363:A:E2 protein:HCV: 0.00: 0.00\n")
        for i in range(0, len(sequence), 50):
            f.write(sequence[i:i+50] + "\n")
        f.write("*\n")
    
    print(f"Created PIR files with sequence length: {len(sequence)}")

def build_models():
    """Build both types of models"""
    
    log.verbose()
    env = Environ()
    env.io.atom_files_directory = ['.', '/home/alexb/Desktop/final/Modele']
    
    # Create PIR files first
    create_pir_files()
    
    # Skip full relaxation since it already worked
    # print("\n=== Building Full Relaxation Model ===")
    # a1 = E2_Model_FullRelax(env,
    #                        alnfile='E2_41_46_full.pir',
    #                        knowns=('E2-m424-429',),
    #                        sequence='E2_target_full')
    # a1.starting_model = 1
    # a1.ending_model = 2
    # a1.make()
    
    print("\n=== Building 28-41 Disulfide Model ===")
    a2 = E2_Model_28_41(env,
                        alnfile='E2_28_41.pir',
                        knowns=('E2-m424-429',),
                        sequence='E2_target_28_41')
    a2.starting_model = 1
    a2.ending_model = 2
    a2.make()
    
    print("\nModeling completed!")
    print("Full relaxation models (41-46): E2_target_full.B99990001.pdb, E2_target_full.B99990002.pdb")
    print("28-41 disulfide models: E2_target_28_41.B99990001.pdb, E2_target_28_41.B99990002.pdb")

if __name__ == "__main__":
    build_models()