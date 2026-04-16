#!/usr/bin/env python3
"""
02_detect_cyx.py

Robust CYX assigner for AMBER/LEaP pipelines.

Key features:
- Accepts disulfide pairs in either:
    * Sequence numbering (e.g. 411, 424), OR
    * PDB numbering (e.g. 28, 41)
- Auto-maps known E2 sequence positions to PDB residue IDs:
    411 -> 28
    424 -> 41
    429 -> 46
- Converts residues involved in disulfides from CYS -> CYX
- Removes thiol hydrogen atoms (HG*, e.g. HG/HG1/HG2) for CYX residues
  to avoid HS/SH parameter errors if an S-S bond exists in the PDB.
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Union, Dict, Set


# ===========================================================
# Default mapping: E2 sequence numbering -> PDB residue IDs
# (validated in your PDBs)
# ===========================================================
DEFAULT_SEQ_TO_PDB: Dict[int, int] = {
    411: 28,
    424: 41,
    429: 46,
}

# Shared disulfides (PDB numbering) - keep as you already had
DEFAULT_SHARED_PAIRS: List[Tuple[int, int]] = [
    (69, 237),
    (76, 103),
    (111, 181),
    (125, 169),
    (186, 214),
    (198, 202),
    (224, 261),
]


# ===========================================================
# IO helpers
# ===========================================================
def load_pdb_lines(path: Path) -> List[str]:
    with open(path, "r") as f:
        return f.readlines()


def write_pdb_lines(path: Path, lines: List[str]) -> None:
    with open(path, "w") as f:
        f.writelines(lines)


# ===========================================================
# PDB parsing helpers
# ===========================================================
def pdb_resids_present(lines: List[str]) -> Set[int]:
    """Return set of residue IDs present in ATOM/HETATM records."""
    resids = set()
    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            resid = int(line[22:26])
        except ValueError:
            continue
        resids.add(resid)
    return resids


def normalize_pairs(
    manual_pairs: Union[List[int], List[Tuple[int, int]], None]
) -> List[Tuple[int, int]]:
    """
    manual_pairs may be:
      - None / [] -> []
      - flat list [a,b,c,d] -> [(a,b),(c,d)]
      - list of tuples [(a,b),...] -> [(a,b),...]
    """
    if not manual_pairs:
        return []

    if isinstance(manual_pairs[0], tuple):
        return [(int(a), int(b)) for (a, b) in manual_pairs]  # type: ignore

    # flat list
    flat = [int(x) for x in manual_pairs]  # type: ignore
    if len(flat) % 2 != 0:
        raise ValueError("SSBOND list must contain an even number of integers")
    it = iter(flat)
    return [(a, b) for a, b in zip(it, it)]


def validate_pairs_no_duplicates(ss_pairs: List[Tuple[int, int]]) -> None:
    """No residue is allowed to appear in more than one disulfide."""
    counts = {}
    for a, b in ss_pairs:
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1
    bad = [r for r, c in counts.items() if c > 1]
    if bad:
        raise RuntimeError(
            f"[FATAL] Invalid disulfide topology: residues appear in multiple bonds: {bad}\n"
            f"Pairs: {ss_pairs}"
        )


def map_pairs_to_pdb_resids(
    ss_pairs: List[Tuple[int, int]],
    present_resids: Set[int],
    seq_to_pdb: Dict[int, int],
) -> List[Tuple[int, int]]:
    """
    If a residue number is not present in the PDB, try mapping via seq_to_pdb.
    This lets you pass in (411,424) and get (28,41) automatically.
    """
    mapped: List[Tuple[int, int]] = []
    missing_raw: Set[int] = set()

    def map_one(r: int) -> int:
        if r in present_resids:
            return r
        if r in seq_to_pdb:
            return seq_to_pdb[r]
        missing_raw.add(r)
        return r  # placeholder

    for a, b in ss_pairs:
        aa = map_one(a)
        bb = map_one(b)
        mapped.append((aa, bb))

    # After mapping, ensure everything exists
    missing_after: Set[int] = set()
    for a, b in mapped:
        if a not in present_resids:
            missing_after.add(a)
        if b not in present_resids:
            missing_after.add(b)

    if missing_raw or missing_after:
        msg = []
        if missing_raw:
            msg.append(f"Unmappable residue numbers (not in PDB and not in mapping): {sorted(missing_raw)}")
        if missing_after:
            msg.append(f"Residues still not present in PDB after mapping: {sorted(missing_after)}")
        raise RuntimeError(
            "[FATAL] Disulfide residue numbering mismatch.\n"
            + "\n".join(msg)
            + "\nFix: pass PDB residue IDs, or extend DEFAULT_SEQ_TO_PDB mapping."
        )

    return mapped


# ===========================================================
# Core CYX assignment
# ===========================================================
def run_detect_cyx(
    input_pdb: Union[str, Path],
    output_pdb: Union[str, Path],
    manual_pairs: Union[List[int], List[Tuple[int, int]], None] = None,
    *,
    seq_to_pdb: Dict[int, int] = None,
    add_shared: bool = False,
) -> List[Tuple[int, int]]:
    """
    Convert CYS -> CYX for residues involved in disulfides and remove HG* on those residues.

    manual_pairs:
      - None / []: no enforced disulfides unless add_shared=True
      - list of tuples: [(411,424), ...] OR [(28,41), ...]
      - flat list: [411,424, 69,237, ...]
    """
    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)

    lines = load_pdb_lines(input_pdb)
    present = pdb_resids_present(lines)

    seq_to_pdb = seq_to_pdb or DEFAULT_SEQ_TO_PDB

    ss_pairs = normalize_pairs(manual_pairs)

    if add_shared:
        ss_pairs = ss_pairs + DEFAULT_SHARED_PAIRS

    print(f"[+] Enforcing disulfide topology → {ss_pairs}")

    # Map pairs to PDB numbering (if needed)
    ss_pairs = map_pairs_to_pdb_resids(ss_pairs, present, seq_to_pdb)

    # Validate topology (no residue in multiple bonds)
    validate_pairs_no_duplicates(ss_pairs)

    residues_in_bonds: Set[int] = {r for pair in ss_pairs for r in pair}
    print(f"[+] CYX residues → {sorted(residues_in_bonds)}")

    # Rewrite: set resname CYX for bonded residues; remove HG* for those residues
    out_lines: List[str] = []
    converted: Set[int] = set()
    removed_hg: Set[int] = set()

    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            out_lines.append(line)
            continue

        atom_name = line[12:16].strip()
        resname = line[17:20].strip()

        try:
            resid = int(line[22:26])
        except ValueError:
            out_lines.append(line)
            continue

        if resid in residues_in_bonds:
            # Drop thiol H on CYX residues to avoid HS/SH issues if S-S bond exists
            # Covers HG, HG1, HG2, etc.
            if atom_name.startswith("HG"):
                removed_hg.add(resid)
                continue

            # Ensure residue name is CYX
            if resname != "CYX":
                # Replace residue name columns 18-20 (0-based 17:20)
                line = line[:17] + "CYX" + line[20:]
                converted.add(resid)

        out_lines.append(line)

    write_pdb_lines(output_pdb, out_lines)

    # Diagnostics
    # Note: converted means we rewrote name to CYX at least once; residues already CYX won't be in converted.
    # We'll check the output file to confirm CYX presence.
    out_check = load_pdb_lines(output_pdb)
    cyx_present: Set[int] = set()
    for l in out_check:
        if not l.startswith(("ATOM", "HETATM")):
            continue
        rn = l[17:20].strip()
        try:
            rid = int(l[22:26])
        except ValueError:
            continue
        if rn == "CYX":
            cyx_present.add(rid)

    missing_cyx = sorted(residues_in_bonds - cyx_present)
    if missing_cyx:
        raise RuntimeError(
            f"[FATAL] Failed to assign CYX to residues: {missing_cyx}\n"
            f"Present residues: (sample) {sorted(list(present))[:30]} ...\n"
            f"Tip: check chain IDs / residue numbering in the input PDB."
        )

    print(f"[✓] CYX detection complete → {output_pdb}")
    if removed_hg:
        print(f"[✓] Removed HG* thiol H on residues → {sorted(removed_hg)}")

    return ss_pairs


# ===========================================================
# CLI entry point
# ===========================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Assign CYX for disulfide-bonded cysteines (numbering-robust).")
    p.add_argument("input", help="Input PDB")
    p.add_argument("output", help="Output PDB")
    p.add_argument("--ssbond", nargs="*", default=[], help="Disulfide pairs as flat ints: a b c d ...")
    p.add_argument("--add-shared", action="store_true", help="Also add default shared disulfides (PDB numbering).")
    args = p.parse_args()

    run_detect_cyx(args.input, args.output, args.ssbond, add_shared=args.add_shared)
