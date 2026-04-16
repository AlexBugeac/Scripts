#!/usr/bin/env python3
"""
add_disulfides.py — Add SSBOND + CONECT records to a PDB file.

Three modes:
  --pairs       explicit list of CYS residue pairs  (46,121 69,243 ...)
  --auto        detect pairs from SG-SG distance (default cutoff 2.5 Å)
  --config      read pairs from a YAML or plain-text file

By default, pairs that exceed --max-distance are skipped unless --force is set.

Usage examples:
  # Explicit pairs
  python add_disulfides.py model.pdb out.pdb --pairs 46,121 69,243 76,104

  # Auto-detect from structure
  python add_disulfides.py model.pdb out.pdb --auto

  # Auto-detect with relaxed cutoff (for freshly-built homology models)
  python add_disulfides.py model.pdb out.pdb --auto --max-distance 6.0

  # Force-add regardless of distance (visualization / MODELLER input)
  python add_disulfides.py model.pdb out.pdb --pairs 46,121 69,243 --force

  # Read pairs from file (one pair per line: "46 121" or "46,121")
  python add_disulfides.py model.pdb out.pdb --config e2_disulfides.txt

  # Override chain ID (default A)
  python add_disulfides.py model.pdb out.pdb --auto --chain B
"""

import argparse
import math
import sys
from pathlib import Path


# ── PDB helpers ───────────────────────────────────────────────────────────────

def parse_atoms(lines: list[str]) -> dict[int, dict]:
    """Return {resnum: {serial, coords, resname, chain}} for all CYS SG atoms."""
    atoms: dict[int, dict] = {}
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        try:
            atom_name = line[12:16].strip()
            res_name  = line[17:20].strip()
            chain     = line[21].strip()
            resnum    = int(line[22:26].strip())
            serial    = int(line[6:11].strip())
            x, y, z   = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except (ValueError, IndexError):
            continue

        if res_name == "CYS" and atom_name == "SG":
            atoms[resnum] = {
                "serial": serial,
                "coords": (x, y, z),
                "chain":  chain,
            }
    return atoms


def sg_distance(a1: dict, a2: dict) -> float:
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(a1["coords"], a2["coords"])))


def find_first_atom_line(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if line.startswith("ATOM"):
            return i
    return len(lines)


# ── Pair sources ──────────────────────────────────────────────────────────────

def pairs_from_cli(raw: list[str]) -> list[tuple[int, int]]:
    """Parse ['46,121', '69,243'] or ['46 121', '69 243'] into tuples."""
    pairs = []
    for token in raw:
        parts = token.replace(",", " ").split()
        if len(parts) != 2:
            sys.exit(f"ERROR: cannot parse pair '{token}' — expected 'R1,R2'")
        pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def pairs_from_config(path: str) -> list[tuple[int, int]]:
    """Read pairs from a plain-text or YAML file.

    Accepted formats:
      plain text:  one pair per line  →  46 121   or   46,121
      YAML list :  - [46, 121]        or   - "46,121"
    """
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: config file '{path}' not found")

    text = p.read_text()

    # Try YAML first if pyyaml available
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
            data = yaml.safe_load(text)
            pairs = []
            for item in data:
                if isinstance(item, (list, tuple)):
                    pairs.append((int(item[0]), int(item[1])))
                elif isinstance(item, str):
                    parts = item.replace(",", " ").split()
                    pairs.append((int(parts[0]), int(parts[1])))
            return pairs
        except ImportError:
            pass  # fall through to plain-text parsing

    # Plain-text
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def pairs_from_auto(cys_atoms: dict[int, dict], cutoff: float) -> list[tuple[int, int]]:
    """Return all CYS-CYS pairs with SG-SG distance ≤ cutoff."""
    residues = sorted(cys_atoms.keys())
    pairs = []
    for i, r1 in enumerate(residues):
        for r2 in residues[i + 1:]:
            if sg_distance(cys_atoms[r1], cys_atoms[r2]) <= cutoff:
                pairs.append((r1, r2))
    return pairs


# ── Main writer ───────────────────────────────────────────────────────────────

def add_disulfides(
    input_pdb:    str,
    output_pdb:   str,
    pairs:        list[tuple[int, int]],
    force:        bool = False,
    max_distance: float = 2.5,
    chain:        str   = "A",
) -> int:
    """
    Write output_pdb with SSBOND + CONECT records for the given CYS pairs.

    Returns the number of bonds actually written.
    """
    lines = Path(input_pdb).read_text().splitlines(keepends=True)
    cys_atoms = parse_atoms(lines)

    print(f"Found {len(cys_atoms)} CYS SG atoms: {sorted(cys_atoms.keys())}")

    ssbond_lines: list[str] = []
    conect_lines: list[str] = []
    bond_num = 0

    for r1, r2 in pairs:
        if r1 not in cys_atoms or r2 not in cys_atoms:
            missing = [str(r) for r in (r1, r2) if r not in cys_atoms]
            print(f"  SKIP  CYS {r1}-{r2}: residue(s) {', '.join(missing)} not found")
            continue

        dist = sg_distance(cys_atoms[r1], cys_atoms[r2])
        ok   = dist <= max_distance or force

        if not ok:
            print(f"  SKIP  CYS {r1:4d}-{r2:4d}: {dist:.2f} Å  (> {max_distance} Å, use --force to override)")
            continue

        bond_num += 1
        flag = "✓" if dist <= 2.5 else "⚠" if dist <= 5.0 else "✗ large"
        print(f"  Bond {bond_num:2d}: CYS {r1:4d}-{r2:4d}  {dist:.2f} Å  {flag}")

        s1 = cys_atoms[r1]["serial"]
        s2 = cys_atoms[r2]["serial"]

        ssbond_lines.append(
            f"SSBOND {bond_num:3d} CYS {chain} {r1:4d}    CYS {chain} {r2:4d}"
            f"                          1555   1555  {dist:.2f}\n"
        )
        conect_lines += [
            f"CONECT{s1:5d}{s2:5d}\n",
            f"CONECT{s2:5d}{s1:5d}\n",
        ]

    # Insert SSBOND records before first ATOM line
    insert = find_first_atom_line(lines)
    lines  = lines[:insert] + ssbond_lines + lines[insert:]

    # Append CONECT records before END (or at file end)
    end_idx = next(
        (i for i, l in enumerate(lines) if l.strip() == "END"),
        len(lines),
    )
    lines = lines[:end_idx] + conect_lines + lines[end_idx:]

    Path(output_pdb).write_text("".join(lines))
    print(f"\nWrote {bond_num} disulfide bond(s) → {output_pdb}")
    return bond_num


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add SSBOND + CONECT records to a PDB file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("input",  help="Input PDB file")
    ap.add_argument("output", help="Output PDB file")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pairs",  nargs="+", metavar="R1,R2",
                     help="Explicit residue pairs, e.g. 46,121 69,243")
    src.add_argument("--auto",   action="store_true",
                     help="Auto-detect pairs by SG-SG distance")
    src.add_argument("--config", metavar="FILE",
                     help="Read pairs from YAML or plain-text file")

    ap.add_argument("--max-distance", type=float, default=2.5,
                    help="Max SG-SG Å to accept (default 2.5; use ~6.0 for homology models)")
    ap.add_argument("--force", action="store_true",
                    help="Write bonds even if distance exceeds --max-distance")
    ap.add_argument("--chain", default="A",
                    help="Chain ID for SSBOND records (default A)")

    args = ap.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"ERROR: input file '{args.input}' not found")

    # Resolve pairs
    if args.pairs:
        pairs = pairs_from_cli(args.pairs)
    elif args.config:
        pairs = pairs_from_config(args.config)
    else:  # --auto
        lines     = Path(args.input).read_text().splitlines(keepends=True)
        cys_atoms = parse_atoms(lines)
        pairs     = pairs_from_auto(cys_atoms, args.max_distance)
        print(f"Auto-detected {len(pairs)} pair(s) within {args.max_distance} Å")

    if not pairs:
        sys.exit("No pairs found — nothing to write.")

    add_disulfides(
        input_pdb    = args.input,
        output_pdb   = args.output,
        pairs        = pairs,
        force        = args.force,
        max_distance = args.max_distance,
        chain        = args.chain,
    )


if __name__ == "__main__":
    main()
