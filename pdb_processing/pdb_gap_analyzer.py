#!/usr/bin/env python3
"""
PDB Gap Analyzer

This script analyzes PDB files to identify missing residues (gaps) in the structure.
Useful for understanding discontinuities in protein structures before homology modeling.

Usage:
    python pdb_gap_analyzer.py structure.pdb
    python pdb_gap_analyzer.py -c A structure.pdb  # Analyze specific chain
    python pdb_gap_analyzer.py -v structure.pdb    # Verbose output
"""

import argparse
import sys
from pathlib import Path
from typing import List, Set, Dict, Tuple


def analyze_pdb_gaps(pdb_file: str, chain_id: str = None, verbose: bool = False) -> Dict:
    """
    Analyze a PDB file for missing residues (gaps).
    
    Args:
        pdb_file: Path to PDB file
        chain_id: Specific chain to analyze (None for all chains)
        verbose: Enable verbose output
        
    Returns:
        Dictionary with gap analysis results
    """
    if not Path(pdb_file).exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    # Store residues by chain
    chains_residues = {}
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                res_num = int(line[22:26])
                chain = line[21].strip()
                
                # Filter by chain if specified
                if chain_id and chain != chain_id:
                    continue
                
                if chain not in chains_residues:
                    chains_residues[chain] = set()
                chains_residues[chain].add(res_num)
    
    if not chains_residues:
        raise ValueError(f"No CA atoms found in PDB file{f' for chain {chain_id}' if chain_id else ''}")
    
    # Analyze gaps for each chain
    results = {}
    
    for chain, residues in chains_residues.items():
        sorted_residues = sorted(residues)
        first_res = sorted_residues[0]
        last_res = sorted_residues[-1]
        total_residues = len(sorted_residues)
        expected_residues = last_res - first_res + 1
        
        # Find gaps
        gaps = []
        missing_residues = []
        current_gap_start = None
        
        for i in range(first_res, last_res + 1):
            if i not in residues:
                missing_residues.append(i)
                if current_gap_start is None:
                    current_gap_start = i
            else:
                if current_gap_start is not None:
                    gaps.append((current_gap_start, i - 1))
                    current_gap_start = None
        
        # Handle gap at the end
        if current_gap_start is not None:
            gaps.append((current_gap_start, last_res))
        
        results[chain] = {
            'first_residue': first_res,
            'last_residue': last_res,
            'total_residues': total_residues,
            'expected_residues': expected_residues,
            'missing_count': len(missing_residues),
            'missing_residues': missing_residues,
            'gaps': gaps,
            'completeness': (total_residues / expected_residues) * 100 if expected_residues > 0 else 0
        }
        
        if verbose:
            print(f"\nChain {chain} Analysis:")
            print(f"  Residue range: {first_res} - {last_res}")
            print(f"  Residues present: {total_residues}")
            print(f"  Expected if continuous: {expected_residues}")
            print(f"  Missing residues: {len(missing_residues)}")
            print(f"  Completeness: {results[chain]['completeness']:.1f}%")
            
            if gaps:
                print(f"  Gap regions:")
                for gap_start, gap_end in gaps:
                    gap_length = gap_end - gap_start + 1
                    print(f"    {gap_start}-{gap_end} ({gap_length} residues)")
            else:
                print("  No gaps found!")
    
    return results


def print_summary(results: Dict, pdb_file: str):
    """Print a summary of the gap analysis."""
    print(f"\nGap Analysis Summary for: {Path(pdb_file).name}")
    print("=" * 50)
    
    total_chains = len(results)
    chains_with_gaps = sum(1 for data in results.values() if data['missing_count'] > 0)
    
    print(f"Total chains analyzed: {total_chains}")
    print(f"Chains with gaps: {chains_with_gaps}")
    
    for chain, data in results.items():
        status = "GAPS PRESENT" if data['missing_count'] > 0 else "COMPLETE"
        print(f"\nChain {chain}: {status}")
        print(f"  Range: {data['first_residue']}-{data['last_residue']} ({data['total_residues']} residues)")
        print(f"  Completeness: {data['completeness']:.1f}%")
        
        if data['gaps']:
            print(f"  Gaps: {len(data['gaps'])} region(s)")
            for gap_start, gap_end in data['gaps']:
                gap_length = gap_end - gap_start + 1
                print(f"    • {gap_start}-{gap_end} ({gap_length} residues)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PDB files for missing residues (gaps)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all chains
  python pdb_gap_analyzer.py structure.pdb
  
  # Analyze specific chain
  python pdb_gap_analyzer.py -c A structure.pdb
  
  # Verbose output with detailed information
  python pdb_gap_analyzer.py -v structure.pdb
  
  # Multiple files
  python pdb_gap_analyzer.py *.pdb
        """
    )
    
    parser.add_argument(
        'pdb_files',
        nargs='+',
        help='PDB file(s) to analyze'
    )
    
    parser.add_argument(
        '-c', '--chain',
        help='Analyze specific chain (default: all chains)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Show only summary, skip detailed output'
    )
    
    args = parser.parse_args()
    
    try:
        for pdb_file in args.pdb_files:
            if len(args.pdb_files) > 1:
                print(f"\n{'=' * 60}")
                print(f"Processing: {pdb_file}")
                print('=' * 60)
            
            results = analyze_pdb_gaps(
                pdb_file, 
                chain_id=args.chain, 
                verbose=args.verbose and not args.summary_only
            )
            
            if not args.verbose or args.summary_only:
                print_summary(results, pdb_file)
            
            # Export gaps for MODELLER alignment if needed
            if any(data['missing_count'] > 0 for data in results.values()):
                gaps_file = Path(pdb_file).stem + "_gaps.txt"
                with open(gaps_file, 'w') as f:
                    f.write(f"# Gap analysis for {pdb_file}\n")
                    for chain, data in results.items():
                        if data['missing_residues']:
                            f.write(f"Chain {chain} missing residues: {','.join(map(str, data['missing_residues']))}\n")
                
                if len(args.pdb_files) == 1:  # Only show message for single file
                    print(f"\nGap information exported to: {gaps_file}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()