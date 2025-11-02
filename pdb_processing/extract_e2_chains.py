#!/usr/bin/env python3
"""
PDB Chain Extraction Script for E2 Protein
Extracts specific chains while preserving disulfide bonds, CONECT records, and metadata
"""

import sys
import re
from pathlib import Path
from collections import defaultdict

def analyze_pdb_chains(input_pdb):
    """
    Analyze PDB file to identify chains and their descriptions
    
    Args:
        input_pdb (str): Path to input PDB file
        
    Returns:
        dict: Chain information with descriptions
    """
    chain_info = {}
    chain_atom_counts = defaultdict(int)
    mol_id_to_desc = {}
    mol_id_to_chains = defaultdict(list)
    
    print(f"Analyzing PDB file: {input_pdb}")
    print("=" * 60)
    
    with open(input_pdb, 'r') as f:
        current_mol_id = None
        current_molecule = None
        current_chains = []
        
        for line in f:
            record_type = line[:6].strip()
            
            # Parse COMPND records to get molecule descriptions
            if record_type == 'COMPND':
                if 'MOL_ID:' in line:
                    # Save previous molecule info
                    if current_mol_id and current_molecule and current_chains:
                        mol_id_to_desc[current_mol_id] = current_molecule
                        mol_id_to_chains[current_mol_id] = current_chains[:]
                    
                    # Extract new MOL_ID
                    match = re.search(r'MOL_ID:\s*(\d+)', line)
                    if match:
                        current_mol_id = int(match.group(1))
                        current_molecule = None
                        current_chains = []
                
                elif 'MOLECULE:' in line and current_mol_id:
                    # Extract molecule name
                    parts = line.split('MOLECULE:')
                    if len(parts) > 1:
                        molecule = parts[1].strip().rstrip(';').strip()
                        current_molecule = molecule
                
                elif 'CHAIN:' in line and current_mol_id:
                    # Extract chain IDs
                    parts = line.split('CHAIN:')
                    if len(parts) > 1:
                        chain_part = parts[1].strip().rstrip(';').strip()
                        # Parse chains (can be like "A, B" or "A,B" or "A")
                        chains = [c.strip() for c in chain_part.replace(',', ' ').split()]
                        current_chains.extend(chains)
            
            # Count atoms per chain
            elif record_type in ['ATOM', 'HETATM']:
                if len(line) > 21:
                    chain_id = line[21]
                    if record_type == 'ATOM':
                        chain_atom_counts[chain_id] += 1
        
        # Save the last molecule info
        if current_mol_id and current_molecule and current_chains:
            mol_id_to_desc[current_mol_id] = current_molecule
            mol_id_to_chains[current_mol_id] = current_chains[:]
    
    # Build chain information
    for mol_id, description in mol_id_to_desc.items():
        for chain in mol_id_to_chains[mol_id]:
            chain_info[chain] = {
                'description': description,
                'mol_id': mol_id,
                'atom_count': chain_atom_counts.get(chain, 0)
            }
    
    # Add any chains that have atoms but weren't in COMPND records
    for chain, count in chain_atom_counts.items():
        if chain not in chain_info:
            chain_info[chain] = {
                'description': 'Unknown',
                'mol_id': None,
                'atom_count': count
            }
    
    return chain_info

def display_chain_info(chain_info):
    """Display chain information in a formatted table"""
    print("\nCHAIN ANALYSIS:")
    print("-" * 80)
    print(f"{'Chain':<6} {'Atoms':<8} {'Mol ID':<8} {'Description':<50}")
    print("-" * 80)
    
    for chain in sorted(chain_info.keys()):
        info = chain_info[chain]
        mol_id = str(info['mol_id']) if info['mol_id'] else 'N/A'
        print(f"{chain:<6} {info['atom_count']:<8} {mol_id:<8} {info['description']:<50}")
    
    print("-" * 80)

def get_user_chain_selection(chain_info):
    """Get chain selection from user input"""
    available_chains = sorted(chain_info.keys())
    
    print(f"\nAvailable chains: {', '.join(available_chains)}")
    print("\nEnter the chains you want to extract:")
    print("  - Single chain: A")
    print("  - Multiple chains: A,B or A B or A,B,C")
    print("  - All chains: all")
    print("  - Quit: q or quit")
    
    while True:
        user_input = input("\nChains to extract: ").strip()
        
        if user_input.lower() in ['q', 'quit']:
            return None
        
        if user_input.lower() == 'all':
            return available_chains
        
        # Parse user input
        if ',' in user_input:
            selected_chains = [c.strip().upper() for c in user_input.split(',')]
        else:
            selected_chains = [c.strip().upper() for c in user_input.split()]
        
        # Validate selection
        invalid_chains = [c for c in selected_chains if c not in available_chains]
        if invalid_chains:
            print(f"Error: Invalid chains: {', '.join(invalid_chains)}")
            print(f"Available chains: {', '.join(available_chains)}")
            continue
        
        if not selected_chains:
            print("Error: No chains selected")
            continue
        
        return selected_chains

def extract_e2_chains(input_pdb, output_pdb, target_chains, remove_hetatm=False):
    """
    Extract specific chains from PDB file while preserving structural information
    
    Args:
        input_pdb (str): Path to input PDB file
        output_pdb (str): Path to output PDB file
        target_chains (list): List of chain IDs to extract
        remove_hetatm (bool): If True, remove HETATM records (NAG, etc.) for cleaner modeling
    """
    # Convert to set for faster lookup
    target_chains_set = set(target_chains)
    
    # Store extracted records
    header_records = []
    atom_records = []
    conect_records = []
    other_records = []
    
    # Keep track of atom numbers that belong to target chains
    target_atom_numbers = set()
    
    print(f"Extracting chains {target_chains} from {input_pdb}")
    
    with open(input_pdb, 'r') as f:
        for line in f:
            record_type = line[:6].strip()
            
            # Keep all header information except chain-specific COMPND
            if record_type in ['HEADER', 'TITLE', 'CAVEAT', 'KEYWDS', 'EXPDTA', 
                             'NUMMDL', 'MDLTYP', 'AUTHOR', 'REVDAT', 'JRNL', 
                             'REMARK', 'DBREF', 'SEQADV', 'SEQRES', 'MODRES',
                             'HELIX', 'SHEET', 'TURN', 'SSBOND', 'CISPEP',
                             'SITE', 'CRYST1', 'ORIGX1', 'ORIGX2', 'ORIGX3',
                             'SCALE1', 'SCALE2', 'SCALE3', 'MTRIX1', 'MTRIX2',
                             'MTRIX3', 'TVECT', 'MODEL', 'ENDMDL']:
                header_records.append(line)
            
            # Handle HET-related records based on remove_hetatm flag
            elif record_type in ['HET', 'HETNAM', 'HETSYN', 'FORMUL']:
                if not remove_hetatm:
                    header_records.append(line)
            
            # Handle LINK records - filter NAG links if removing HETATM
            elif record_type == 'LINK':
                if remove_hetatm and ('NAG' in line or any(het in line for het in ['BMA', 'MAN', 'FUC', 'GAL'])):
                    # Skip glycosylation LINK records when removing HETATM
                    continue
                else:
                    header_records.append(line)
            
            # Filter COMPND records to only include E2-related entries
            elif record_type == 'COMPND':
                # Keep E2-related COMPND records
                if 'E2' in line or any(f'CHAIN: {chain}' in line or f'{chain},' in line for chain in target_chains):
                    header_records.append(line)
                # Also keep general MOL_ID records
                elif 'MOL_ID:' in line:
                    header_records.append(line)
            
            # Filter SOURCE records similarly
            elif record_type == 'SOURCE':
                # Keep E2-related SOURCE records (look for MOL_ID: 2 which corresponds to E2)
                if ('MOL_ID: 2' in line or 'HCV' in line or 'S52' in line or 
                    'HEPACIVIRUS' in line or any(f'CHAIN: {chain}' in line for chain in target_chains)):
                    header_records.append(line)
            
            # Extract ATOM and HETATM records for target chains
            elif record_type in ['ATOM', 'HETATM']:
                if len(line) > 21:  # Ensure line is long enough
                    chain_id = line[21]  # Chain ID is at position 21
                    if chain_id in target_chains_set:
                        # Skip HETATM records if remove_hetatm is True
                        if record_type == 'HETATM' and remove_hetatm:
                            continue
                        
                        atom_records.append(line)
                        # Extract atom number for CONECT record filtering
                        try:
                            atom_num = int(line[6:11].strip())
                            target_atom_numbers.add(atom_num)
                        except ValueError:
                            pass
            
            # Keep ANISOU records for target chains
            elif record_type == 'ANISOU':
                if len(line) > 21:
                    chain_id = line[21]
                    if chain_id in target_chains_set:
                        atom_records.append(line)
            
            # Filter CONECT records to only include connections within target chains
            elif record_type == 'CONECT':
                conect_records.append(line)
            
            # Keep TER records for target chains
            elif record_type == 'TER':
                if len(line) > 21:
                    chain_id = line[21]
                    if chain_id in target_chains_set:
                        atom_records.append(line)
            
            # Keep END record
            elif record_type == 'END':
                other_records.append(line)
    
    # Filter CONECT records to only include atoms from target chains
    filtered_conect_records = []
    for line in conect_records:
        parts = line.split()
        if len(parts) >= 2:
            try:
                # Check if the first atom number is in our target set
                first_atom = int(parts[1])
                if first_atom in target_atom_numbers:
                    # Check if all connected atoms are also in target set
                    all_in_target = True
                    for i in range(2, len(parts)):
                        try:
                            atom_num = int(parts[i])
                            if atom_num not in target_atom_numbers:
                                all_in_target = False
                                break
                        except ValueError:
                            break
                    
                    if all_in_target:
                        filtered_conect_records.append(line)
            except ValueError:
                continue
    
    # Write output file
    print(f"Writing extracted structure to {output_pdb}")
    with open(output_pdb, 'w') as f:
        # Write header records
        for record in header_records:
            f.write(record)
        
        # Write atom records
        for record in atom_records:
            f.write(record)
        
        # Write filtered CONECT records
        for record in filtered_conect_records:
            f.write(record)
        
        # Write other records (like END)
        for record in other_records:
            f.write(record)
    
    # Report statistics
    atom_count = len([r for r in atom_records if r.startswith('ATOM')])
    hetatm_count = len([r for r in atom_records if r.startswith('HETATM')])
    conect_count = len(filtered_conect_records)
    
    print(f"Extraction complete!")
    print(f"  Chains extracted: {target_chains}")
    print(f"  ATOM records: {atom_count}")
    if not remove_hetatm:
        print(f"  HETATM records: {hetatm_count}")
    else:
        print(f"  HETATM records: {hetatm_count} (removed: NAG glycosylations)")
    print(f"  CONECT records: {conect_count}")
    print(f"  Target atoms tracked: {len(target_atom_numbers)}")
    
    if remove_hetatm and hetatm_count == 0:
        print("  ✓ All HETATM molecules successfully removed for clean modeling")

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Usage: python extract_e2_chains.py <input_pdb> [output_pdb] [chain1,chain2,...]")
        print("Example: python extract_e2_chains.py 8RJJ.pdb")
        print("Example: python extract_e2_chains.py 8RJJ.pdb 8RJJ_E2_only.pdb B,D")
        sys.exit(1)
    
    input_pdb = sys.argv[1]
    
    # Check if input file exists
    if not Path(input_pdb).exists():
        print(f"Error: Input file {input_pdb} not found!")
        sys.exit(1)
    
    # Analyze chains in the PDB file
    chain_info = analyze_pdb_chains(input_pdb)
    display_chain_info(chain_info)
    
    # Get target chains from command line or user input
    target_chains = None
    output_pdb = None
    
    if len(sys.argv) >= 4:
        # Command line mode: input.pdb output.pdb chains
        output_pdb = sys.argv[2]
        target_chains = [chain.strip().upper() for chain in sys.argv[3].split(',')]
    elif len(sys.argv) == 3:
        # Check if second argument is output file or chains
        second_arg = sys.argv[2]
        if '.' in second_arg and not ',' in second_arg:
            # Likely an output filename
            output_pdb = second_arg
            target_chains = get_user_chain_selection(chain_info)
        else:
            # Likely chain specification
            target_chains = [chain.strip().upper() for chain in second_arg.split(',')]
    else:
        # Interactive mode
        target_chains = get_user_chain_selection(chain_info)
    
    # Exit if user quit
    if target_chains is None:
        print("Extraction cancelled.")
        sys.exit(0)
    
    # Validate selected chains
    available_chains = set(chain_info.keys())
    invalid_chains = [c for c in target_chains if c not in available_chains]
    if invalid_chains:
        print(f"Error: Invalid chains specified: {', '.join(invalid_chains)}")
        print(f"Available chains: {', '.join(sorted(available_chains))}")
        sys.exit(1)
    
    # Generate output filename if not provided
    if output_pdb is None:
        input_path = Path(input_pdb)
        chain_suffix = "_".join(target_chains)
        output_pdb = str(input_path.parent / f"{input_path.stem}_{chain_suffix}_only{input_path.suffix}")
    
    # Show what will be extracted
    print(f"\nEXTRACTION PLAN:")
    print("-" * 40)
    print(f"Input file: {input_pdb}")
    print(f"Output file: {output_pdb}")
    print(f"Chains to extract: {', '.join(target_chains)}")
    print("\nSelected chains:")
    for chain in target_chains:
        info = chain_info[chain]
        print(f"  {chain}: {info['description']} ({info['atom_count']} atoms)")
    
    # Ask about HETATM removal
    print(f"\nHETATM molecules (NAG, etc.) found in structure.")
    print("For homology modeling, it's recommended to remove these.")
    print("For MD simulations, keeping them is more biologically accurate.")
    remove_hetatm = input(f"Remove HETATM molecules? (Y/n): ").strip().lower()
    remove_hetatm = remove_hetatm not in ['n', 'no']
    
    if remove_hetatm:
        print("✓ Will remove HETATM molecules (NAG glycosylations, etc.)")
    else:
        print("✓ Will keep HETATM molecules")
    
    # Confirm extraction
    confirm = input(f"\nProceed with extraction? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Extraction cancelled.")
        sys.exit(0)
    
    # Extract chains
    extract_e2_chains(input_pdb, output_pdb, target_chains, remove_hetatm)

if __name__ == "__main__":
    main()