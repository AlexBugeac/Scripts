#!/usr/bin/env python

import argparse
from modeller import *
from modeller.scripts import complete_pdb

# -------------------------
# Parse arguments
# -------------------------
parser = argparse.ArgumentParser(
    description="Add ACE/NME terminal caps using MODELLER."
)

parser.add_argument("-i", "--input", required=True,
                    help="Input PDB file")
parser.add_argument("-o", "--output", required=True,
                    help="Output PDB file")
parser.add_argument("--caps", action="store_true",
                    help="Add ACE to N-terminus and NME to C-terminus")

args = parser.parse_args()

# -------------------------
# MODELLER environment
# -------------------------
log.verbose()
env = Environ()

env.io.hetatm = True
env.io.water = True

env.libs.topology.read(file='$(LIB)/top_heav.lib')
env.libs.parameters.read(file='$(LIB)/par.lib')

print(f"Loading PDB: {args.input}")
mdl = complete_pdb(env, args.input)

# -------------------------
# Add ACE/NME caps
# -------------------------
if args.caps:
    first = mdl.residues[0]     # first residue by index
    last  = mdl.residues[-1]    # last residue

    print(f"Adding ACE to residue {first.resid}")
    mdl.patch(residue_type='ACE', residues=first)

    print(f"Adding NME to residue {last.resid}")
    mdl.patch(residue_type='NME', residues=last)

# -------------------------
# Save output
# -------------------------
mdl.write(file=args.output)
print(f"Saved repaired model as {args.output}")
