#!/usr/bin/env python3
"""
Compute Cα–Cα distance between two residues (0-based residue index).

- Consistent with previous scripts (MDTraj residue index)
- Processes traj.dcd, traj2.dcd, traj3.dcd, traj4.dcd
- Saves residue info for verification
- Outputs distance in Å
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mdtraj as md

ALLOWED_SEGS = ["traj.dcd", "traj2.dcd", "traj3.dcd", "traj4.dcd"]

def find_ca_atom(topology, res_index):
    residue = topology.residue(res_index)
    for atom in residue.atoms:
        if atom.name == "CA":
            return atom.index
    raise RuntimeError(f"No CA atom found in residue index {res_index}")

def list_segments(replica_dir: Path):
    segs = []
    for name in ALLOWED_SEGS:
        p = replica_dir / name
        if p.exists():
            segs.append(p)
    return segs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system_name", required=True)
    ap.add_argument("--replica_dir", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--res1", type=int, required=True)
    ap.add_argument("--res2", type=int, required=True)
    ap.add_argument("--dt_fs", type=float, default=2.0)
    ap.add_argument("--report_steps", type=int, default=5000)
    ap.add_argument("--chunk_frames", type=int, default=5000)
    ap.add_argument("--outdir", default="analysis_out_distance")
    args = ap.parse_args()

    replica_dir = Path(args.replica_dir).expanduser().resolve()
    top_path = Path(args.top).expanduser().resolve()

    outdir = replica_dir.parent / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    segs = list_segments(replica_dir)
    if not segs:
        raise RuntimeError("No DCD segments found.")

    # Load reference frame
    ref = next(md.iterload(str(segs[0]), top=str(top_path), chunk=1))
    top = ref.topology

    if args.res1 >= top.n_residues or args.res2 >= top.n_residues:
        raise ValueError("Residue index out of range.")

    atom1 = find_ca_atom(top, args.res1)
    atom2 = find_ca_atom(top, args.res2)

    res1_obj = top.residue(args.res1)
    res2_obj = top.residue(args.res2)

    print("\n=== DISTANCE INFO ===")
    print(f"Residue 1: index0={args.res1}, resSeq={res1_obj.resSeq}, name={res1_obj.name}, CA_atom={atom1}")
    print(f"Residue 2: index0={args.res2}, resSeq={res2_obj.resSeq}, name={res2_obj.name}, CA_atom={atom2}")
    print("=====================\n")

    ps_per_frame = (args.dt_fs * args.report_steps) / 1000.0

    rep_name = replica_dir.name
    out_csv = outdir / f"{args.system_name}_{rep_name}_dist_res{args.res1}_{args.res2}.csv"

    if out_csv.exists():
        out_csv.unlink()

    global_frame = 0

    for seg_i, seg_path in enumerate(segs, start=1):
        print(f"Processing {seg_path.name}")
        for chunk in md.iterload(str(seg_path), top=str(top_path), chunk=args.chunk_frames):
            n = chunk.n_frames
            frames = np.arange(global_frame, global_frame + n, dtype=int)
            time_ps = frames.astype(float) * ps_per_frame

            dist_nm = md.compute_distances(chunk, [[atom1, atom2]])[:, 0]
            dist_A = dist_nm * 10.0

            df = pd.DataFrame({
                "system": args.system_name,
                "replica": rep_name,
                "segment": seg_path.name,
                "frame_global": frames,
                "time_ps": time_ps,
                "distance_A": dist_A,
                "res1_index0": args.res1,
                "res1_resSeq": res1_obj.resSeq,
                "res1_name": res1_obj.name,
                "res2_index0": args.res2,
                "res2_resSeq": res2_obj.resSeq,
                "res2_name": res2_obj.name
            })

            df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            global_frame += n

    print("\nDone.")
    print("Output:", out_csv)

if __name__ == "__main__":
    main()
