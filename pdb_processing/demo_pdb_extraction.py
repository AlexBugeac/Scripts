#!/usr/bin/env python3
"""
Demo script showing how to use the PDB chain extraction tool
"""

import subprocess
import sys
from pathlib import Path

def demo_extraction():
    """Demonstrate the chain extraction process"""
    script_path = Path(__file__).parent / "extract_e2_chains.py"
    
    print("PDB Chain Extraction Tool - Demo")
    print("=" * 50)
    print()
    print("This tool can extract specific chains from PDB files while preserving:")
    print("  ✓ Disulfide bonds (SSBOND records)")
    print("  ✓ Connectivity (CONECT records)")
    print("  ✓ Structural metadata")
    print("  ✓ Chain-specific annotations")
    print()
    
    print("Usage examples:")
    print("1. Interactive mode (analyze chains first):")
    print(f"   python {script_path.name} input.pdb")
    print()
    print("2. Specify output file interactively:")
    print(f"   python {script_path.name} input.pdb output.pdb")
    print()
    print("3. Command line mode (direct extraction):")
    print(f"   python {script_path.name} input.pdb output.pdb A,B")
    print()
    
    # Check if we have any PDB files to demo with
    template_dir = Path.home() / "Desktop" / "final" / "Templates"
    if template_dir.exists():
        pdb_files = list(template_dir.glob("*.pdb"))
        if pdb_files:
            print("Available PDB files for testing:")
            for pdb_file in pdb_files[:5]:  # Show first 5
                print(f"  {pdb_file}")
            print()
            
            # Ask user if they want to run a demo
            response = input("Would you like to run a demo with one of these files? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                print(f"\nRunning: python {script_path} {pdb_files[0]}")
                subprocess.run([sys.executable, str(script_path), str(pdb_files[0])])

if __name__ == "__main__":
    demo_extraction()