#!/usr/bin/env python3

import sys

if len(sys.argv) < 2:
    print("Usage: python pdb_categories.py file.pdb")
    sys.exit(1)

filename = sys.argv[1]
categories = set()

with open(filename, "r") as f:
    for line in f:
        if len(line) < 6:
            continue
        record = line[:6].strip()
        if record:  # ignore blank lines
            categories.add(record)

print("Categories found in PDB:")
for c in sorted(categories):
    print("  " + c)
