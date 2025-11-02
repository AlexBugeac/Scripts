#!/usr/bin/env python3

"""
Script to automatically detect and add disulfide bonds to a PDB structure using MODELLER
Based on proximity of cysteine SG atoms
"""

from modeller import environ, model, forms

# Initialize environment
env = environ()
env.io.atom_files_directory = ['.']

def add_disulfide_bonds_by_proximity(input_pdb, output_pdb, distance_cutoff=3.0, force_bond=None, exclude_residues=None):
    """
    Read a PDB file and automatically add disulfide bonds based on proximity
    
    Args:
        input_pdb (str): Path to input PDB file
        output_pdb (str): Path to output PDB file with disulfide bonds
        distance_cutoff (float): Maximum distance for disulfide bond (Angstroms)
        force_bond (tuple): Tuple of (res1, res2) to force a specific disulfide bond
        exclude_residues (list): List of residue numbers to exclude from automatic bonding
    """
    
    # Read the structure
    mdl = model(env, file=input_pdb)
    
    # Find all cysteine residues
    cysteines = []
    for residue in mdl.residues:
        if residue.name == 'CYS':
            cysteines.append(residue)
    
    print(f"Found {len(cysteines)} cysteine residues")
    
    # Find SG atoms and calculate distances
    disulfide_bonds = []
    
    # If force_bond is specified, add it first
    if force_bond:
        res1_num, res2_num = force_bond
        cys1 = None
        cys2 = None
        
        # Find the specified residues
        for residue in cysteines:
            if residue.num == str(res1_num):
                cys1 = residue
            elif residue.num == str(res2_num):
                cys2 = residue
        
        if cys1 and cys2:
            # Find SG atoms
            sg1 = next((atom for atom in cys1.atoms if atom.name == 'SG'), None)
            sg2 = next((atom for atom in cys2.atoms if atom.name == 'SG'), None)
            
            if sg1 and sg2:
                distance = ((sg1.x - sg2.x)**2 + (sg1.y - sg2.y)**2 + (sg1.z - sg2.z)**2)**0.5
                disulfide_bonds.append((cys1, cys2, distance))
                print(f"FORCED Disulfide bond: {cys1.name}{cys1.num} - {cys2.name}{cys2.num}, distance: {distance:.2f} Å")
    
    # Find automatic bonds (excluding specified residues)
    for i in range(len(cysteines)):
        for j in range(i+1, len(cysteines)):
            cys1 = cysteines[i]
            cys2 = cysteines[j]
            
            # Skip if either residue should be excluded or already in a forced bond
            if exclude_residues:
                if int(cys1.num) in exclude_residues or int(cys2.num) in exclude_residues:
                    continue
            
            # Skip if this pair is already in disulfide_bonds (from forced bonds)
            already_bonded = any((bond[0] == cys1 and bond[1] == cys2) or 
                               (bond[0] == cys2 and bond[1] == cys1) 
                               for bond in disulfide_bonds)
            if already_bonded:
                continue
            
            # Find SG atoms
            sg1 = None
            sg2 = None
            
            for atom in cys1.atoms:
                if atom.name == 'SG':
                    sg1 = atom
                    break
            
            for atom in cys2.atoms:
                if atom.name == 'SG':
                    sg2 = atom
                    break
            
            if sg1 and sg2:
                # Calculate distance
                distance = ((sg1.x - sg2.x)**2 + (sg1.y - sg2.y)**2 + (sg1.z - sg2.z)**2)**0.5
                
                if distance <= distance_cutoff:
                    disulfide_bonds.append((cys1, cys2, distance))
                    print(f"Disulfide bond: {cys1.name}{cys1.num} - {cys2.name}{cys2.num}, distance: {distance:.2f} Å")
    
    # Create disulfide bonds
    if disulfide_bonds:
        print(f"\nCreating {len(disulfide_bonds)} disulfide bonds...")
        
        # Change CYS to CYS for bonded cysteines (keep as CYS, don't change to CYX)
        bonded_cysteines = set()
        for bond in disulfide_bonds:
            bonded_cysteines.add(bond[0])
            bonded_cysteines.add(bond[1])
        
        # Note: We keep them as CYS, not CYX
        
        # Add the actual disulfide bonds
        for i, (cys1, cys2, distance) in enumerate(disulfide_bonds):
            # Find SG atoms again
            sg1 = next(atom for atom in cys1.atoms if atom.name == 'SG')
            sg2 = next(atom for atom in cys2.atoms if atom.name == 'SG')
            
            # Create the bond (this is MODELLER-specific syntax)
            mdl.restraints.add(forms.bond(sg1, sg2))
    
    # Write the structure with disulfide bonds
    mdl.write(file=output_pdb, model_format='PDB')
    print(f"\nStructure with disulfide bonds written to: {output_pdb}")
    
    return disulfide_bonds

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python add_disulfide_bonds.py input.pdb output.pdb")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # For your specific case: force 41-46 bond and exclude residue 46 from other bonds
    # (except for the forced bond)
    bonds = add_disulfide_bonds_by_proximity(
        input_file, 
        output_file,
        force_bond=(41, 46),
        exclude_residues=[46]  # Prevent 46 from bonding to anything except the forced bond
    )
    
    print(f"\nSummary:")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Disulfide bonds created: {len(bonds)}")