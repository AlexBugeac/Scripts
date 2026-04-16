#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_time_axis(log_paths: list[Path]) -> pd.DataFrame | None:
    """
    Best-effort load time/step per logged row from log.csv/log2.csv.
    Returns dataframe with columns: log_row, time_ps, step
    """
    for lp in log_paths:
        if not lp.exists():
            continue
        try:
            df = pd.read_csv(lp)
        except Exception:
            continue

        cols = {c.lower().strip(): c for c in df.columns}
        time_col = None
        step_col = None

        for key in ("time (ps)", "time_ps", "time", "t (ps)"):
            if key in cols:
                time_col = cols[key]
                break
        for key in ("step", "steps"):
            if key in cols:
                step_col = cols[key]
                break

        out = pd.DataFrame()
        out["log_row"] = range(len(df))
        out["time_ps"] = df[time_col] if time_col else pd.NA
        out["step"] = df[step_col] if step_col else pd.NA
        return out

    return None


def count_frames_mdtraj(top: Path, dcd: Path, chunk: int = 5000) -> int:
    import mdtraj as md

    n = 0
    for part in md.iterload(str(dcd), top=str(top), chunk=chunk):
        n += part.n_frames
    return n


def maybe_merge_dcd(top: Path, dcds: list[Path], out_dcd: Path) -> None:
    """
    WARNING: loads whole trajectories into RAM (mdtraj limitation).
    """
    import mdtraj as md

    trajs = [md.load(str(d), top=str(top)) for d in dcds]
    merged = trajs[0].join(trajs[1:]) if len(trajs) > 1 else trajs[0]
    merged.save_dcd(str(out_dcd))


def main():
    ap = argparse.ArgumentParser(
        description="Build a per-frame trajectory index CSV from one PRMTOP and any number of DCDs."
    )
    ap.add_argument("-t", "--top", required=True, type=Path, help="Topology PRMTOP file")
    ap.add_argument(
        "-d", "--dcd", required=True, nargs="+", type=Path,
        help="One or more DCD files (will be concatenated in the order given)"
    )
    ap.add_argument(
        "-o", "--out", default=None, type=Path,
        help="Output CSV path. Default: <top_basename>__index.csv in current dir"
    )
    ap.add_argument(
        "--tag", default=None,
        help="Optional tag stored in CSV (e.g., 8RJJ-native_replica01)"
    )
    ap.add_argument(
        "--log", default=[], nargs="*", type=Path,
        help="Optional log CSV(s) to map time/step (e.g., log.csv log2.csv)."
    )
    ap.add_argument(
        "--merge-dcd", action="store_true",
        help="Also write merged DCD (RAM heavy)."
    )
    ap.add_argument(
        "--merged-out", default=None, type=Path,
        help="Merged DCD output path. Default: <out_csv_stem>__merged.dcd"
    )
    ap.add_argument(
        "--chunk", default=5000, type=int,
        help="Chunk size for mdtraj.iterload when counting frames (default: 5000)"
    )

    args = ap.parse_args()

    top = args.top.expanduser().resolve()
    if not top.exists():
        raise FileNotFoundError(f"PRMTOP not found: {top}")

    dcds = [p.expanduser().resolve() for p in args.dcd]
    for d in dcds:
        if not d.exists():
            raise FileNotFoundError(f"DCD not found: {d}")

    out_csv = args.out
    if out_csv is None:
        out_csv = Path.cwd() / f"{top.stem}__trajectory_index.csv"
    out_csv = out_csv.expanduser().resolve()

    # Count frames per DCD
    frame_counts = []
    for d in dcds:
        n = count_frames_mdtraj(top, d, chunk=args.chunk)
        frame_counts.append(n)

    # Build index rows
    rows = []
    global_frame = 0
    for dcd_path, n_frames in zip(dcds, frame_counts):
        for sf in range(n_frames):
            rows.append(
                {
                    "global_frame": global_frame,
                    "source_file": dcd_path.name,
                    "source_path": str(dcd_path),
                    "source_frame": sf,
                }
            )
            global_frame += 1

    df = pd.DataFrame(rows)

    # Optional tag
    if args.tag:
        df["tag"] = args.tag

    # Best-effort time mapping from logs
    log_paths = args.log if args.log else []
    log_df = load_time_axis(log_paths) if log_paths else None
    if log_df is not None:
        minlen = min(len(df), len(log_df))
        df.loc[: minlen - 1, "time_ps"] = log_df.loc[: minlen - 1, "time_ps"].values
        df.loc[: minlen - 1, "step"] = log_df.loc[: minlen - 1, "step"].values
    else:
        df["time_ps"] = pd.NA
        df["step"] = pd.NA

    # Write CSV
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv} (frames={len(df)})")
    print(f"     top: {top}")
    print(f"     dcds: {[d.name for d in dcds]}")
    if args.tag:
        print(f"     tag: {args.tag}")

    # Optional merged DCD
    if args.merge_dcd:
        merged_out = args.merged_out
        if merged_out is None:
            merged_out = out_csv.with_suffix("").with_name(out_csv.stem + "__merged.dcd")
        merged_out = merged_out.expanduser().resolve()
        maybe_merge_dcd(top, dcds, merged_out)
        print(f"[OK] wrote merged DCD -> {merged_out}")


if __name__ == "__main__":
    main()
