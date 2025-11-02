#!/usr/bin/env python3
"""
Forced Template Substitution Approach
Creates a hybrid alignment where 6MEJ sequence literally replaces 8RJJ/8RK0 in region 519-536
"""

def create_forced_substitution_alignment():
    """Create alignment with forced 6MEJ substitution in target region"""
    
    # Read the original alignment
    with open('/home/alexb/Desktop/final/E2_hybrid_alignment.pir', 'r') as f:
        content = f.read()
    
    print("Original alignment length:")
    lines = content.strip().split('\n')
    for i, line in enumerate(lines):
        if line.startswith('>P1;'):
            print(f"  {line}")
            if i+2 < len(lines):
                seq_line = lines[i+2]
                print(f"  Sequence length: {len(seq_line.replace('*', ''))}")
    
    # Parse sequences
    sequences = {}
    current_id = None
    for line in lines:
        if line.startswith('>P1;'):
            current_id = line[4:]  # Remove '>P1;'
            sequences[current_id] = {'header': '', 'sequence': ''}
        elif current_id and line and not line.startswith('>'):
            if ':' in line and sequences[current_id]['header'] == '':
                sequences[current_id]['header'] = line
            else:
                sequences[current_id]['sequence'] += line.replace('*', '')
    
    print("\nParsed sequences:")
    for seq_id, data in sequences.items():
        print(f"  {seq_id}: {len(data['sequence'])} residues")
    
    # Target region in alignment coordinates
    # We need to find positions 519-536 in the target sequence
    target_seq = sequences['E2_target']['sequence']
    print(f"\nTarget sequence length: {len(target_seq)}")
    
    # The region 519-536 corresponds to positions 135-152 in model (0-based: 134-151)
    region_start = 134  # 0-based
    region_end = 152    # 0-based, exclusive
    
    print(f"Target region {region_start+1}-{region_end} in alignment:")
    print(f"  Original: {target_seq[region_start:region_end]}")
    
    # Get 6MEJ sequence for this region
    mej_seq = sequences['6MEJ_0001']['sequence']
    print(f"  6MEJ region: {mej_seq[region_start:region_end]}")
    
    # Create modified sequences where 8RJJ and 8RK0 use 6MEJ sequence in target region
    modified_sequences = {}
    for seq_id, data in sequences.items():
        if seq_id in ['8RJJ_0001', '8RK0_0001']:
            # Replace region with 6MEJ sequence
            orig_seq = data['sequence']
            modified_seq = (orig_seq[:region_start] + 
                          mej_seq[region_start:region_end] + 
                          orig_seq[region_end:])
            modified_sequences[seq_id] = {
                'header': data['header'],
                'sequence': modified_seq
            }
            print(f"\nModified {seq_id}:")
            print(f"  Original region: {orig_seq[region_start:region_end]}")
            print(f"  Modified region: {modified_seq[region_start:region_end]}")
        else:
            modified_sequences[seq_id] = data
    
    # Write new alignment
    output_file = '/home/alexb/Desktop/final/E2_forced_substitution_alignment.pir'
    with open(output_file, 'w') as f:
        for seq_id, data in modified_sequences.items():
            f.write(f">P1;{seq_id}\n")
            f.write(f"{data['header']}\n")
            f.write(f"{data['sequence']}*\n")
    
    print(f"\nForced substitution alignment written to: {output_file}")
    return output_file

if __name__ == "__main__":
    create_forced_substitution_alignment()