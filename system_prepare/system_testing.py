#!/usr/bin/env python3
from openmm.app import AmberPrmtopFile, AmberInpcrdFile
import sys

if len(sys.argv) != 3:
    print("Usage: python test_openmm_load.py file.prmtop file.rst7")
    sys.exit(1)

prmtop_file = sys.argv[1]
inpcrd_file = sys.argv[2]

print("Testing OpenMM load...")
print(f"PRMTOP : {prmtop_file}")
print(f"RST7   : {inpcrd_file}")
print("-" * 40)

# Load PRMTOP
try:
    prmtop = AmberPrmtopFile(prmtop_file)
    print("✓ PRMTOP loaded successfully")
except Exception as e:
    print("✗ ERROR loading PRMTOP:")
    print(e)
    sys.exit(1)

# Load RST7
try:
    inpcrd = AmberInpcrdFile(inpcrd_file)
    print("✓ RST7 loaded successfully")
    print(f"Positions shape: {len(inpcrd.positions)} atoms")
except Exception as e:
    print("✗ ERROR loading RST7:")
    print(e)
    sys.exit(1)

print("-" * 40)
print("All good! OpenMM can load your files.")
