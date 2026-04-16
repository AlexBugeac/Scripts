#!/usr/bin/env python

import argparse
from modeller import *
from modeller.scripts import complete_pdb

parser = argparse.ArgumentParser(
    description="Repair PDB with MODELLER: rebuild topology + DISU patches (no caps)."
)
parser.add_argument("-i", "--input", required=True, help="Input PDB file")
parser.add_argument("-o", "--output", required=True, help="Output repaired PDB file")
args = parser.parse_args()

log.verbose()
env = environ()
env.io.hetatm = True
env.io.water = True

# Topology & parameters
env.libs.topology.read(file='$(LIB)/top_heav.lib')
env.libs.parameters.read(file='$(LIB)/par.lib')

# Load model
mdl = complete_pdb(env, args.input)

# MODELLER-renumbered disulfide pairs (chain A)
ss_pairs = [
    (26, 100),
    (49, 217),
    (56, 83),
    (91, 161),
    (105, 149),
    (166, 194),
    (178, 182),
    (204, 241),
]

print("Using MODELLER-renumbered SS bonds:")
for a, b in ss_pairs:
    print(f"  {a} <--> {b}")
    mdl.patch(
        residue_type='DISU',
        residues=(mdl.residues[f"{a}:A"], mdl.residues[f"{b}:A"])
    )

# Write output
mdl.write(file=args.output)
print(f"Done. Wrote: {args.output}")
