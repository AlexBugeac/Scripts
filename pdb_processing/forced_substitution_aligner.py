#!/usr/bin/env python3
"""
forced_substitution_aligner.py — Replace a region in a PIR alignment.

Takes a donor template sequence and substitutes its residues into one or more
recipient template sequences over a specified residue window. Useful for
hybrid modeling where you want a specific template's conformation in one region
but another template's conformation everywhere else.

Usage examples:
  # Replace positions 519-536 in 8RJJ and 8RK0 with the 6MEJ sequence
  python forced_substitution_aligner.py \\
    --input    alignment.pir \\
    --output   hybrid_alignment.pir \\
    --donor    6MEJ_0001 \\
    --targets  8RJJ_0001 8RK0_0001 \\
    --start    519 \\
    --end      536

  # Use 0-based alignment column indices instead of residue numbers
  python forced_substitution_aligner.py \\
    --input alignment.pir --output hybrid.pir \\
    --donor 6MEJ_0001 --targets 8RJJ_0001 \\
    --col-start 134 --col-end 152
"""

import argparse
import sys
from pathlib import Path


# ── PIR I/O ───────────────────────────────────────────────────────────────────

def parse_pir(text: str) -> list[dict]:
    """Return list of {id, header, sequence} dicts preserving order."""
    entries = []
    current = None
    for line in text.splitlines():
        if line.startswith(">P1;"):
            current = {"id": line[4:].strip(), "header": "", "sequence": ""}
            entries.append(current)
        elif current is not None:
            if not current["header"] and (":" in line or line.strip() == ""):
                current["header"] = line
            else:
                current["sequence"] += line.replace("*", "").strip()
    return entries


def write_pir(entries: list[dict]) -> str:
    lines = []
    for e in entries:
        lines.append(f">P1;{e['id']}")
        lines.append(e["header"])
        lines.append(e["sequence"] + "*")
    return "\n".join(lines) + "\n"


def resnum_to_col(sequence: str, resnum: int) -> int:
    """Convert 1-based residue number to 0-based column index (gaps count)."""
    # Assumes no gaps in non-aligned PIR sequences; for gapped alignments
    # the column IS the position directly.
    return resnum - 1


# ── Main logic ────────────────────────────────────────────────────────────────

def substitute(args) -> None:
    text    = Path(args.input).read_text()
    entries = parse_pir(text)
    by_id   = {e["id"]: e for e in entries}

    # Validate IDs
    missing = [i for i in [args.donor] + args.targets if i not in by_id]
    if missing:
        sys.exit(f"ERROR: sequence ID(s) not found in alignment: {missing}\n"
                 f"Available: {list(by_id.keys())}")

    donor_seq = by_id[args.donor]["sequence"]

    # Resolve column range
    if args.col_start is not None:
        col_lo = args.col_start
        col_hi = args.col_end
    else:
        col_lo = resnum_to_col(donor_seq, args.start)
        col_hi = resnum_to_col(donor_seq, args.end) + 1  # exclusive

    donor_region = donor_seq[col_lo:col_hi]
    print(f"Donor   : {args.donor}")
    print(f"Region  : columns {col_lo}–{col_hi - 1} ({col_hi - col_lo} residues)")
    print(f"Donor region sequence: {donor_region}")

    for target_id in args.targets:
        orig = by_id[target_id]["sequence"]
        if len(orig) < col_hi:
            print(f"  WARNING: {target_id} sequence length {len(orig)} < col_hi {col_hi} — skipping")
            continue
        modified = orig[:col_lo] + donor_region + orig[col_hi:]
        by_id[target_id]["sequence"] = modified
        print(f"  {target_id}: [{orig[col_lo:col_hi]}] → [{donor_region}]")

    Path(args.output).write_text(write_pir(entries))
    print(f"\nWritten → {args.output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replace a region in PIR alignment sequences with a donor sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input",   required=True, help="Input PIR alignment file")
    ap.add_argument("--output",  required=True, help="Output PIR alignment file")
    ap.add_argument("--donor",   required=True, help="Sequence ID to copy region FROM")
    ap.add_argument("--targets", required=True, nargs="+",
                    help="Sequence ID(s) to substitute region INTO")

    loc = ap.add_mutually_exclusive_group(required=True)
    loc.add_argument("--start", type=int, metavar="RESNUM",
                     help="1-based start residue number (uses donor sequence for reference)")
    loc.add_argument("--col-start", type=int, metavar="COL",
                     help="0-based alignment column index start")

    ap.add_argument("--end",     type=int, metavar="RESNUM",
                    help="1-based end residue number (inclusive, required with --start)")
    ap.add_argument("--col-end", type=int, metavar="COL",
                    help="0-based alignment column index end (exclusive, required with --col-start)")

    args = ap.parse_args()

    if args.start is not None and args.end is None:
        ap.error("--end is required when using --start")
    if args.col_start is not None and args.col_end is None:
        ap.error("--col-end is required when using --col-start")
    if not Path(args.input).exists():
        sys.exit(f"ERROR: input file '{args.input}' not found")

    substitute(args)


if __name__ == "__main__":
    main()
