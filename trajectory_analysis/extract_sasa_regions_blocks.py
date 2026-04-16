#!/usr/bin/env python3
"""
extract_sasa_regions_blocks.py

Purpose:
  Extract ONLY SASA-derived metrics from OpenMM DCD segments (traj.dcd, traj2.dcd, traj3.dcd, traj4.dcd)
  for:
    (A) user-defined residue-index ranges (e.g. 8-19,20-60,115-130)
    (B) automatic blocks of size N residues (default 20): 0-19,20-39,40-59,...

Numbering (IMPORTANT — matches your previous analysis):
  - Ranges are specified in RESIDUE INDEX (0-based) as used by MDTraj topology.residue(i)
  - This is the same convention you used for dist_res1/dist_res2 earlier (e.g., 8 and 19 -> GLN/ASN)

Outputs (per replica):
  - CSV time series with per-frame:
      frame_global, time_ps, segment, segment_index,
      sasa_total_nm2,
      sasa_region_<start>_<end>_nm2 for each user range,
      sasa_block_<start>_<end>_nm2 for each auto block.
  - Units: nm^2 (Å^2 = nm^2 * 100)

Example:
  python extract_sasa_regions_blocks.py \
    --system_name 8RJJ-native \
    --replica_dir replica_01 \
    --top ~/AlexSimulations/E2_HCV_REDO/Simulation_files_final/8RJJ-native/8RJJ-native_05_minimized.prmtop \
    --dt_fs 2 --report_steps 5000 \
    --regions "8-19,20-60,115-130" \
    --block_size 20 \
    --chunk_frames 2000

Notes:
  - Computes Shrake-Rupley in mode="residue", then sums residue SASA for each region/block.
  - Only reads the DCDs; does not require coordinate CSVs.
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import mdtraj as md


ALLOWED_SEGS = ["traj.dcd", "traj2.dcd", "traj3.dcd", "traj4.dcd"]


def parse_regions(s: str):
    """
    "8-19,20-60,115-130" -> [(8,19),(20,60),(115,130)]
    Inclusive endpoints.
    """
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", part)
        if not m:
            raise ValueError(f"Bad region format: '{part}' (expected like 8-19)")
        a, b = int(m.group(1)), int(m.group(2))
        if b < a:
            a, b = b, a
        out.append((a, b))
    return out


def list_segments(replica_dir: Path):
    segs = []
    for name in ALLOWED_SEGS:
        p = replica_dir / name
        if p.exists():
            segs.append(p)
    return segs


def build_blocks(n_residues: int, block_size: int):
    """
    Residue-index blocks over [0..n_residues-1], inclusive endpoints.
    block_size=20 -> (0,19),(20,39),...
    """
    blocks = []
    start = 0
    while start < n_residues:
        end = min(start + block_size - 1, n_residues - 1)
        blocks.append((start, end))
        start += block_size
    return blocks


def append_csv(df: pd.DataFrame, outpath: Path):
    exists = outpath.exists()
    df.to_csv(outpath, index=False, mode="a", header=not exists)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system_name", required=True)
    ap.add_argument("--replica_dir", required=True, help="Path to replica_01 / replica_02 / replica_03")
    ap.add_argument("--top", required=True, help="PRMTOP (preferred) or PDB topology")

    ap.add_argument("--dt_fs", type=float, default=2.0)
    ap.add_argument("--report_steps", type=int, default=5000)

    ap.add_argument("--regions", default="", help='Comma list of residue-index ranges, e.g. "8-19,20-60,115-130"')
    ap.add_argument("--block_size", type=int, default=20, help="Auto block size in residues (default 20)")

    ap.add_argument("--chunk_frames", type=int, default=2000, help="Frames per iterload chunk (default 2000)")
    ap.add_argument("--sasa_sphere_points", type=int, default=480, help="Shrake-Rupley n_sphere_points (default 480)")

    ap.add_argument("--outdir", default="analysis_out_sasa", help="Output dir (created at system root)")
    args = ap.parse_args()

    replica_dir = Path(args.replica_dir).expanduser().resolve()
    if not replica_dir.exists():
        raise FileNotFoundError(f"replica_dir not found: {replica_dir}")

    system_root = replica_dir.parent
    outdir = system_root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    top_path = Path(args.top).expanduser().resolve()
    if not top_path.exists():
        raise FileNotFoundError(f"top not found: {top_path}")

    segs = list_segments(replica_dir)
    if not segs:
        raise RuntimeError(f"No segments found in {replica_dir}. Expected any of: {', '.join(ALLOWED_SEGS)}")

    user_regions = parse_regions(args.regions)

    # time conversion
    ps_per_frame = (args.dt_fs * args.report_steps) / 1000.0  # fs*steps -> fs, /1000 -> ps

    # output filename per replica
    rep_name = replica_dir.name  # replica_01
    out_csv = outdir / f"{args.system_name}_{rep_name}_sasa_regions_blocks_firstpass.csv"
    if out_csv.exists():
        out_csv.unlink()  # start fresh each run

    # Load ref just to know n_residues
    ref = next(md.iterload(str(segs[0]), top=str(top_path), chunk=1))
    n_res = ref.topology.n_residues

    # Validate user regions within residue index space
    for a, b in user_regions:
        if a < 0 or b >= n_res:
            raise ValueError(f"Region {a}-{b} out of bounds for topology residues (0..{n_res-1}).")

    blocks = build_blocks(n_res, args.block_size)

    print("\n=== SASA REGION/BLOCK EXTRACTION ===")
    print(f"System        : {args.system_name}")
    print(f"Replica       : {rep_name}")
    print(f"Topology      : {top_path}")
    print(f"Segments      : {[p.name for p in segs]}")
    print(f"Residues      : {n_res} (indices 0..{n_res-1})")
    print(f"ps/frame      : {ps_per_frame}")
    print(f"User regions  : {user_regions if user_regions else 'None'}")
    print(f"Block size    : {args.block_size} -> {len(blocks)} blocks")
    print(f"Chunk frames  : {args.chunk_frames}")
    print(f"Sphere points : {args.sasa_sphere_points}")
    print(f"Output CSV    : {out_csv}")

    global_frame = 0

    # Pre-build column names for speed
    region_cols = [f"sasa_region_{a}_{b}_nm2" for (a, b) in user_regions]
    block_cols  = [f"sasa_block_{a}_{b}_nm2" for (a, b) in blocks]

    for seg_i, seg_path in enumerate(segs, start=1):
        print(f"\n--- Segment {seg_i}: {seg_path.name} ---")
        for chunk in md.iterload(str(seg_path), top=str(top_path), chunk=args.chunk_frames):
            n = chunk.n_frames
            frames = np.arange(global_frame, global_frame + n, dtype=np.int64)
            time_ps = frames.astype(np.float64) * ps_per_frame

            # residue SASA: (n_frames, n_residues), nm^2
            sasa_res_nm2 = md.shrake_rupley(
                chunk,
                mode="residue",
                n_sphere_points=args.sasa_sphere_points
            )

            # total SASA per frame
            sasa_total_nm2 = sasa_res_nm2.sum(axis=1)

            # compute region sums
            region_data = {}
            for (a, b), col in zip(user_regions, region_cols):
                region_data[col] = sasa_res_nm2[:, a:b+1].sum(axis=1)

            # compute block sums
            block_data = {}
            for (a, b), col in zip(blocks, block_cols):
                block_data[col] = sasa_res_nm2[:, a:b+1].sum(axis=1)

            df = pd.DataFrame({
                "system": args.system_name,
                "replica": rep_name,
                "segment": seg_path.name,
                "segment_index": seg_i,
                "frame_global": frames,
                "time_ps": time_ps,
                "sasa_total_nm2": sasa_total_nm2,
                **region_data,
                **block_data,
            })

            append_csv(df, out_csv)

            global_frame += n
            print(f"  wrote frames {int(frames[0])}..{int(frames[-1])} (total={global_frame})")

    print("\nDone.")
    print("Wrote:", out_csv)
    print("Units: nm^2 (Å^2 = nm^2 * 100)")


if __name__ == "__main__":
    main()
