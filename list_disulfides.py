#!/usr/bin/env python3
import argparse, math

parser = argparse.ArgumentParser(description="Detect disulfide bonds by distance between SG atoms (no SSBONDs required)")
parser.add_argument("pdb", help="Path to PDB file")
parser.add_argument("--cutoff", type=float, default=2.2,
                    help="Max distance (Å) between SG atoms to consider a disulfide (default: 2.2 Å)")
args = parser.parse_args()

# Collect SG atom coordinates
sg_atoms = []
with open(args.pdb) as f:
    for line in f:
        if line.startswith("ATOM") and line[13:16].strip() == "SG" and line[17:20].strip() == "CYS":
            chain = line[21].strip() or "A"
            resid = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            sg_atoms.append((resid, chain, (x, y, z)))

def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

pairs = []
for i, (r1, c1, xyz1) in enumerate(sg_atoms):
    for j, (r2, c2, xyz2) in enumerate(sg_atoms):
        if j <= i:
            continue
        d = dist(xyz1, xyz2)
        if d <= args.cutoff:
            pairs.append((r1, c1, r2, c2, d))

print(f"📜 Disulfides detected in {args.pdb} (distance ≤ {args.cutoff} Å):")
if not pairs:
    print("   (No close SG–SG pairs found)")
else:
    for i, (r1, c1, r2, c2, d) in enumerate(pairs, 1):
        print(f"{i:2d}. {r1}:{c1} — {r2}:{c2}   ({d:.2f} Å)")
    print("\n💡 MODELLER arguments:")
    for r1, c1, r2, c2, _ in pairs:
        print(f"--ss {r1}:{c1}-{r2}:{c2}", end=" ")
    print()
