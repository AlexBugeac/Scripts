#!/usr/bin/env python3
"""
PDB Sequence Extractor

This script extracts amino acid sequences from PDB files, including gaps for missing residues.
Useful for creating PIR alignment files for homology modeling with MODELLER.

Usage:
    python pdb_sequence_extractor.py structure.pdb
    python pdb_sequence_extractor.py -c A structure.pdb     # Extract specific chain
    python pdb_sequence_extractor.py --gaps structure.pdb   # Include gaps as dashes
    python pdb_sequence_extractor.py --pir structure.pdb    # Output in PIR format
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PDBSequenceExtractor:
    """Extract amino acid sequences from PDB files."""
    
    # Standard amino acid three-to-one letter conversion
    AA_MAP = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        # Non-standard amino acids
        'MSE': 'M',  # Selenomethionine
        'SEC': 'U',  # Selenocysteine
        'PYL': 'O',  # Pyrrolysine
    }
    
    def __init__(self, pdb_file: str):
        """Initialize with PDB file path."""
        self.pdb_file = Path(pdb_file)
        if not self.pdb_file.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    def extract_sequences(self, chain_id: Optional[str] = None, include_gaps: bool = False) -> Dict[str, Dict]:
        """
        Extract amino acid sequences from PDB file.
        
        Args:
            chain_id: Specific chain to extract (None for all chains)
            include_gaps: Include gaps as dashes for missing residues
            
        Returns:
            Dictionary with chain sequences and metadata
        """
        chains_data = {}
        
        # First pass: collect all residues
        with open(self.pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    res_num = int(line[22:26])
                    chain = line[21].strip()
                    res_name = line[17:20].strip()
                    
                    # Filter by chain if specified
                    if chain_id and chain != chain_id:
                        continue
                    
                    if chain not in chains_data:
                        chains_data[chain] = {}
                    
                    chains_data[chain][res_num] = self.AA_MAP.get(res_name, 'X')
        
        if not chains_data:
            raise ValueError(f"No CA atoms found{f' for chain {chain_id}' if chain_id else ''}")
        
        # Second pass: create sequences
        results = {}
        
        for chain, residues in chains_data.items():
            sorted_res_nums = sorted(residues.keys())
            first_res = sorted_res_nums[0]
            last_res = sorted_res_nums[-1]
            
            if include_gaps:
                # Include gaps for missing residues
                sequence = ''
                for i in range(first_res, last_res + 1):
                    if i in residues:
                        sequence += residues[i]
                    else:
                        sequence += '-'
            else:
                # Continuous sequence without gaps
                sequence = ''.join(residues[res_num] for res_num in sorted_res_nums)
            
            results[chain] = {
                'sequence': sequence,
                'first_residue': first_res,
                'last_residue': last_res,
                'length': len(sequence),
                'gaps': include_gaps and '-' in sequence,
                'gap_count': sequence.count('-') if include_gaps else 0
            }
        
        return results
    
    def format_fasta(self, sequences: Dict[str, Dict], description: str = "") -> str:
        """Format sequences as FASTA."""
        fasta_output = ""
        pdb_code = self.pdb_file.stem
        
        for chain, data in sequences.items():
            header = f">{pdb_code}_{chain}"
            if description:
                header += f" {description}"
            header += f" | Chain {chain} | Residues {data['first_residue']}-{data['last_residue']}"
            if data.get('gaps'):
                header += f" | {data['gap_count']} gaps"
            
            fasta_output += header + "\n"
            
            # Format sequence in 60-character lines
            sequence = data['sequence']
            for i in range(0, len(sequence), 60):
                fasta_output += sequence[i:i+60] + "\n"
            fasta_output += "\n"
        
        return fasta_output
    
    def format_pir(self, sequences: Dict[str, Dict], sequence_type: str = "structure", 
                   description: str = "", resolution: float = 2.00, r_factor: float = -1.00) -> str:
        """Format sequences as PIR for MODELLER."""
        pir_output = ""
        pdb_code = self.pdb_file.stem
        
        for chain, data in sequences.items():
            # PIR header
            header = f">P1;{pdb_code}_{chain}"
            pir_output += header + "\n"
            
            # PIR description line
            if sequence_type == "structure":
                desc_line = f"structureX:{pdb_code}_{chain}:{data['first_residue']}:{chain}:{data['last_residue']}:{chain}:{description}:{resolution:.2f}:{r_factor:.2f}"
            else:
                desc_line = f"sequence:{pdb_code}_{chain}:{data['first_residue']}:{chain}:{data['last_residue']}:{chain}:{description}:{resolution:.2f}:{r_factor:.2f}"
            
            pir_output += desc_line + "\n"
            
            # Sequence with PIR terminator
            sequence = data['sequence'] + "*"
            pir_output += sequence + "\n\n"
        
        return pir_output


def main():
    parser = argparse.ArgumentParser(
        description="Extract amino acid sequences from PDB files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all chains as FASTA
  python pdb_sequence_extractor.py structure.pdb
  
  # Extract specific chain with gaps
  python pdb_sequence_extractor.py -c A --gaps structure.pdb
  
  # Output in PIR format for MODELLER
  python pdb_sequence_extractor.py --pir structure.pdb
  
  # Save to file
  python pdb_sequence_extractor.py structure.pdb > sequences.fasta
  
  # Multiple files
  python pdb_sequence_extractor.py *.pdb
        """
    )
    
    parser.add_argument(
        'pdb_files',
        nargs='+',
        help='PDB file(s) to process'
    )
    
    parser.add_argument(
        '-c', '--chain',
        help='Extract specific chain (default: all chains)'
    )
    
    parser.add_argument(
        '--gaps',
        action='store_true',
        help='Include gaps as dashes for missing residues'
    )
    
    parser.add_argument(
        '--pir',
        action='store_true',
        help='Output in PIR format for MODELLER'
    )
    
    parser.add_argument(
        '--description',
        default="",
        help='Description to include in headers'
    )
    
    parser.add_argument(
        '--resolution',
        type=float,
        default=2.00,
        help='Resolution for PIR format (default: 2.00)'
    )
    
    parser.add_argument(
        '--r-factor',
        type=float,
        default=-1.00,
        help='R-factor for PIR format (default: -1.00)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file (default: stdout)'
    )
    
    parser.add_argument(
        '--sequence-type',
        choices=['structure', 'sequence'],
        default='structure',
        help='PIR sequence type (default: structure)'
    )
    
    args = parser.parse_args()
    
    try:
        all_output = ""
        
        for pdb_file in args.pdb_files:
            extractor = PDBSequenceExtractor(pdb_file)
            sequences = extractor.extract_sequences(
                chain_id=args.chain,
                include_gaps=args.gaps
            )
            
            if args.pir:
                output = extractor.format_pir(
                    sequences,
                    sequence_type=args.sequence_type,
                    description=args.description,
                    resolution=args.resolution,
                    r_factor=args.r_factor
                )
            else:
                output = extractor.format_fasta(sequences, args.description)
            
            if len(args.pdb_files) > 1:
                all_output += f"# File: {pdb_file}\n"
            all_output += output
            
            # Print summary to stderr if outputting to stdout
            if not args.output:
                total_chains = len(sequences)
                total_residues = sum(data['length'] for data in sequences.values())
                gaps_present = any(data.get('gaps', False) for data in sequences.values())
                
                print(f"# Processed {pdb_file}: {total_chains} chain(s), {total_residues} total residues", 
                      file=sys.stderr)
                if gaps_present:
                    print(f"# Gaps included as dashes", file=sys.stderr)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                f.write(all_output)
            print(f"Sequences written to: {args.output}", file=sys.stderr)
        else:
            print(all_output, end='')
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()