#!/usr/bin/env python3
from pathlib import Path

def run_trim(input_pdb, output_pdb, trim_ranges):
    """
    Trim residues based on ranges and return residue-number mapping:
    old_resnum -> new_resnum

    If residue X was deleted, it is not present in mapping.
    """
    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)

    # -----------------------------
    # Parse trim ranges (1-21, etc.)
    # -----------------------------
    trim_resnums = set()
    for r in trim_ranges:
        a, b = r.split("-")
        a = int(a); b = int(b)
        for x in range(a, b + 1):
            trim_resnums.add(x)

    allowed_cyx_h = {"H", "HA"}

    out = []
    mapping = {}       # old → new numbering
    new_resnum = 1     # renumber residues after trimming

    last_old = None    # track when the residue changes

    with open(input_pdb) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                out.append(line)
                continue

            old_resname = line[17:20].strip()
            old_resnum  = int(line[22:26])
            atom        = line[12:16].strip()

            # skip whole residues inside trimming ranges
            if old_resnum in trim_resnums:
                continue

            # CYX hydrogen cleanup
            if old_resname == "CYX":
                if atom.startswith("H") and atom not in allowed_cyx_h:
                    continue

            # detect new residue start
            if old_resnum != last_old:
                mapping[old_resnum] = new_resnum
                last_old = old_resnum
                new_resnum += 1

            # write new residue number into the PDB line
            new_line = (
                line[:22] +
                f"{mapping[old_resnum]:4d}" +
                line[26:]
            )
            out.append(new_line)

    with open(output_pdb, "w") as f:
        f.writelines(out)

    print(f"[✓] Trim + Renumber complete → {output_pdb}")
    print(f"[Info] Residues kept: {len(mapping)}")

    # return mapping for SSBOND remapping
    return mapping


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_pdb")
    p.add_argument("output_pdb")
    p.add_argument("--trim", nargs="*", required=True)
    args = p.parse_args()

    run_trim(args.input_pdb, args.output_pdb, args.trim)
