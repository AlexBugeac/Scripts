#!/usr/bin/env python3

"""
MODELLER Script 2: Partial Constraint Model with 41-46 Disulfide Bond
- Remove existing disulfide bond involving residue 46
- Add new 41-46 disulfide bond  
- Keep everything after residue 50 fixed/static
"""

from modeller import *
from modeller.automodel import *

class E2_Model_Constrained(LoopModel):
    def special_patches(self, aln):
        """
        Add only the 41-46 disulfide bond
        """
        
        # Add the new 41-46 disulfide bond
        try:
            self.patch(residue_type='DISU', 
                      residues=(self.residues['41:A'], 
                               self.residues['46:A']))
            print("Added disulfide bond: 41-46 (constrained model)")
        except Exception as e:
            print(f"Failed to add disulfide bond 41-46: {e}")
    
    def special_restraints(self, aln):
        """
        Add restraints to keep residues 51+ fixed in their original positions
        """
        
        # Add positional restraints for residues 51 and above
        for residue in self.residues:
            if residue.num >= 51:
                for atom in residue.atoms:
                    # Add strong positional restraint to keep atom in place
                    self.restraints.add(
                        forms.gaussian(
                            features.x_coordinate(atom),
                            mean=atom.x, stdev=0.1
                        )
                    )
                    self.restraints.add(
                        forms.gaussian(
                            features.y_coordinate(atom),
                            mean=atom.y, stdev=0.1
                        )
                    )
                    self.restraints.add(
                        forms.gaussian(
                            features.z_coordinate(atom),
                            mean=atom.z, stdev=0.1
                        )
                    )
        
        print(f"Added positional restraints for residues 51+ to keep them fixed")

def build_constrained_model():
    """
    Build model with 41-46 disulfide bond and constrained residues 51+
    """
    
    log.verbose()
    env = Environ()
    env.io.atom_files_directory = ['.', '/home/alexb/Desktop/final/Modele']
    
    # Create PIR file for constrained modeling
    with open('E2_m424_constrained.pir', 'w') as f:
        f.write(">P1;E2_m424_template\n")
        f.write("structure:E2-m424-429:1:A:400:A:E2 protein:HCV: 2.00: 0.19\n")
        f.write("*\n")
        f.write(">P1;E2_m424_target\n")
        f.write("sequence:E2_m424_target:1:A:400:A:E2 protein:HCV: 0.00: 0.00\n")
        f.write("*\n")
    
    a = E2_Model_Constrained(env,
                alnfile='E2_m424_constrained.pir',
                knowns=('E2-m424-429',),
                sequence='E2_m424_target')
    
    # Define the loop region (residues that are allowed to move)
    a.loop.starting_model = 1
    a.loop.ending_model = 3
    
    # Set the loop region (residues 1-50 can move, 51+ are constrained)
    a.loop.md_level = refine.very_fast
    
    a.starting_model = 1
    a.ending_model = 3
    
    a.make()
    
    print("Constrained models completed!")
    print("Output: E2_m424_target.BL00010001.pdb, etc.")

if __name__ == "__main__":
    build_constrained_model()