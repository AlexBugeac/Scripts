#!/usr/bin/env python3
"""
Simple PDB file downloader from RCSB Protein Data Bank.
Downloads PDB files in parallel for improved performance.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def download_pdb(pdb_id, outdir, overwrite=False):
    """Download a single PDB file from RCSB."""
    pdb_id = pdb_id.strip().upper()
    if not pdb_id:
        return None
    
    filepath = Path(outdir) / f"{pdb_id}.pdb"
    
    # Skip if file exists and not overwriting
    if filepath.exists() and not overwrite:
        return f"⚠️  {pdb_id}: Already exists, skipping"
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        filepath.write_text(response.text)
        return f"✅ {pdb_id}: Downloaded successfully"
        
    except requests.exceptions.RequestException as e:
        return f"❌ {pdb_id}: Failed to download - {e}"
    except Exception as e:
        return f"❌ {pdb_id}: Error saving file - {e}"


def read_pdb_ids(listfile):
    """Read and validate PDB IDs from file."""
    try:
        with open(listfile) as f:
            pdb_ids = [line.strip().upper() for line in f if line.strip()]
        
        if not pdb_ids:
            print("❌ No valid PDB IDs found in file")
            return []
            
        # Basic validation: PDB IDs should be 4 characters
        valid_ids = []
        for pdb_id in pdb_ids:
            if len(pdb_id) == 4 and pdb_id.isalnum():
                valid_ids.append(pdb_id)
            else:
                print(f"⚠️  Skipping invalid PDB ID: {pdb_id}")
        
        return valid_ids
        
    except FileNotFoundError:
        print(f"❌ File not found: {listfile}")
        return []
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Download PDB files from RCSB Protein Data Bank",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s pdb_list.txt
  %(prog)s pdb_list.txt -o structures --overwrite --parallel 10
        """
    )
    parser.add_argument("listfile", help="File containing PDB IDs (one per line)")
    parser.add_argument("-o", "--outdir", default="pdbs", 
                       help="Output directory (default: pdbs)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing files")
    parser.add_argument("--parallel", type=int, default=5, metavar="N",
                       help="Number of parallel downloads (default: 5)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.parallel < 1:
        print("❌ Parallel downloads must be at least 1")
        sys.exit(1)
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    
    # Read PDB IDs
    pdb_ids = read_pdb_ids(args.listfile)
    if not pdb_ids:
        sys.exit(1)
    
    print(f"📁 Output directory: {outdir}")
    print(f"📋 Found {len(pdb_ids)} valid PDB IDs")
    print(f"🚀 Using {args.parallel} parallel downloads")
    print("-" * 50)
    
    # Download files in parallel
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all download tasks
        future_to_pdb = {
            executor.submit(download_pdb, pdb_id, outdir, args.overwrite): pdb_id 
            for pdb_id in pdb_ids
        }
        
        # Process completed downloads
        for future in as_completed(future_to_pdb):
            result = future.result()
            if result:
                print(result)
                if result.startswith("✅"):
                    success_count += 1
    
    print("-" * 50)
    print(f"📊 Summary: {success_count}/{len(pdb_ids)} files downloaded successfully")


if __name__ == "__main__":
    main()
