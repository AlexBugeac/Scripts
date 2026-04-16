#!/usr/bin/env python3
"""
analyze_replica_segments.py

Chunk-safe MDTraj analysis for OpenMM DCD segments.

Goals (per replica):
- Use ONLY these trajectory segments if present, in this order:
    traj.dcd, traj2.dcd, traj3.dcd, traj4.dcd
  (ignores traj_stride*, *_sXX.dcd, etc.)
- Treat segments as ONE continuous trajectory (global frame index).
- Output CSVs you can plot from:
    1) timeseries metrics (per frame): RMSD, distance(res8CA-res19CA), Rg, total SASA
    2) RMSF (Cα) per residue over ALL frames (combined)
    3) Mean per-residue SASA over ALL frames (combined)  [optional]
- Works with older MDTraj by using md.iterload (no slice frame loading).
- Stream-writes the timeseries CSV to avoid huge RAM usage.

Units:
- RMSD, distances, Rg in nm by default (MDTraj native). (You can convert to Å later: *10)
- SASA in nm^2 by default (MDTraj). (Å^2 = nm^2 * 100)

Defaults match your OpenMM run script:
- dt_fs = 2.0
- report_steps = 5000
=> dt_ps per saved frame = 10.0 ps

Example:
python analyze_replica_segments.py \
  --system_name 8RJJ-native \
  --replica_dir replica_01 \
  --top /path/to/system_05_minimized.prmtop \
  --dt_fs 2 --report_steps 5000 \
  --align ca \
  --dist_res1 8 --dist_res2 19 \
  --sasa_per_res_mean yes
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md


ALLOWED_SEGS = ["traj.dcd", "traj2.dcd", "traj3.dcd", "traj4.dcd"]


def list_segments(replica_dir: Path):
    segs = []
    for name in ALLOWED_SEGS:
        p = replica_dir / name
        if p.exists():
            segs.append(p)
    return segs


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def append_csv(df: pd.DataFrame, path: Path):
    exists = path.exists()
    df.to_csv(path, index=False, mode="a", header=not exists)


def ca_indices(top: md.Topology):
    idx = top.select("protein and name CA")
    if idx.size == 0:
        raise RuntimeError("No protein Cα atoms found (selection: 'protein and name CA').")
    return idx


def align_indices(top: md.Topology, align_mode: str):
    if align_mode == "ca":
        idx = top.select("protein and name CA")
    elif align_mode == "backbone":
        # MDTraj backbone includes N, CA, C, O
        idx = top.select("protein and backbone")
    elif align_mode == "none":
        idx = None
    else:
        raise ValueError(f"Unknown align mode: {align_mode}")
    if align_mode != "none" and (idx is None or idx.size == 0):
        raise RuntimeError(f"Alignment selection returned 0 atoms for mode={align_mode}")
    return idx


def atom_index_from_res(top: md.Topology, res_index0: int, atom_name: str = "CA") -> int:
    res = top.residue(int(res_index0))
    for a in res.atoms:
        if a.name == atom_name:
            return a.index
    raise RuntimeError(f"Residue res_index={res_index0} ({res.name}) has no atom named {atom_name}")


def residue_table_for_ca(top: md.Topology, ca_idx: np.ndarray) -> pd.DataFrame:
    rows = []
    for a in ca_idx:
        atom = top.atom(int(a))
        res = atom.residue
        resseq = getattr(res, "resSeq", None)
        rows.append(
            {
                "res_index": res.index,  # 0-based
                "resSeq": int(resseq) if resseq is not None else res.index,
                "resname": res.name,
                "ca_atom_index": int(a),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system_name", default=None, help="Label used in output filenames (default: parent folder name)")
    ap.add_argument("--replica_dir", required=True, help="Path to replica_XX directory")
    ap.add_argument("--top", required=True, help="Topology file: PRMTOP preferred (or PDB)")

    ap.add_argument("--outdir", default="analysis_out", help="Output directory (created at system root)")
    ap.add_argument("--dt_fs", type=float, default=2.0, help="Integrator timestep in fs (default 2)")
    ap.add_argument("--report_steps", type=int, default=5000, help="Steps between saved frames (default 5000)")
    ap.add_argument("--align", choices=["ca", "backbone", "none"], default="ca", help="Alignment/RMSD atom set")

    # Your requested distance: GLN(8) - ASN(19) by res_index (0-based)
    ap.add_argument("--dist_res1", type=int, default=8, help="Residue index (0-based) for distance endpoint 1 (default 8)")
    ap.add_argument("--dist_res2", type=int, default=19, help="Residue index (0-based) for distance endpoint 2 (default 19)")
    ap.add_argument("--dist_atom", default="CA", help="Atom name within those residues (default CA)")

    # SASA options
    ap.add_argument("--sasa_total", choices=["yes", "no"], default="yes", help="Compute total SASA per frame (default yes)")
    ap.add_argument("--sasa_per_res_mean", choices=["yes", "no"], default="no",
                    help="Also compute mean per-residue SASA over all frames (default no)")

    # Optional extra distance (e.g. disulfide SG-SG)
    ap.add_argument("--extra_dist_res1", type=int, default=None, help="Optional: extra distance residue1 res_index (0-based)")
    ap.add_argument("--extra_dist_atom1", default="SG", help="Optional: extra distance atom1 name (default SG)")
    ap.add_argument("--extra_dist_res2", type=int, default=None, help="Optional: extra distance residue2 res_index (0-based)")
    ap.add_argument("--extra_dist_atom2", default="SG", help="Optional: extra distance atom2 name (default SG)")

    # Performance
    ap.add_argument("--chunk_frames", type=int, default=5000, help="Frames per chunk (default 5000)")
    ap.add_argument("--sasa_sphere_points", type=int, default=480, help="Shrake-Rupley n_sphere_points (default 480)")

    args = ap.parse_args()

    replica_dir = Path(args.replica_dir).resolve()
    if not replica_dir.exists():
        raise FileNotFoundError(f"replica_dir not found: {replica_dir}")

    system_root = replica_dir.parent
    system_name = args.system_name if args.system_name else system_root.name

    outdir = system_root / args.outdir
    ensure_dir(outdir)

    top_path = Path(os.path.expanduser(args.top)).resolve()
    if not top_path.exists():
        raise FileNotFoundError(f"top not found: {top_path}")

    segs = list_segments(replica_dir)
    if not segs:
        raise RuntimeError(
            f"No allowed segments found in {replica_dir}. Expected any of: {', '.join(ALLOWED_SEGS)}"
        )

    # Time conversion: fs * steps -> fs; /1000 -> ps
    ps_per_frame = (args.dt_fs * args.report_steps) / 1000.0

    # Prepare outputs (one per replica)
    rep_name = replica_dir.name  # replica_01
    timeseries_csv = outdir / f"{system_name}_{rep_name}_metrics.csv"
    rmsf_csv = outdir / f"{system_name}_{rep_name}_rmsf_ca.csv"
    sasa_res_mean_csv = outdir / f"{system_name}_{rep_name}_sasa_per_res_mean.csv"

    # Start fresh each run
    for p in [timeseries_csv, rmsf_csv, sasa_res_mean_csv]:
        if p.exists():
            p.unlink()

    # Load reference (first frame of first segment)
    ref = next(md.iterload(str(segs[0]), top=str(top_path), chunk=1))
    top = ref.topology

    ca_idx = ca_indices(top)
    align_idx = align_indices(top, args.align)

    # Distance endpoints (res-based)
    dist_a = atom_index_from_res(top, args.dist_res1, args.dist_atom)
    dist_b = atom_index_from_res(top, args.dist_res2, args.dist_atom)

    # Optional extra distance endpoints
    extra_pair = None
    if args.extra_dist_res1 is not None and args.extra_dist_res2 is not None:
        e1 = atom_index_from_res(top, args.extra_dist_res1, args.extra_dist_atom1)
        e2 = atom_index_from_res(top, args.extra_dist_res2, args.extra_dist_atom2)
        extra_pair = (e1, e2, args.extra_dist_res1, args.extra_dist_atom1, args.extra_dist_res2, args.extra_dist_atom2)

    # RMSF streaming accumulators (CA coords after alignment)
    sum_xyz = np.zeros((ca_idx.size, 3), dtype=np.float64)
    sum_xyz2 = np.zeros((ca_idx.size, 3), dtype=np.float64)
    total_frames = 0

    # Mean per-residue SASA accumulators (optional)
    do_sasa_total = (args.sasa_total == "yes")
    do_sasa_res_mean = (args.sasa_per_res_mean == "yes")
    if do_sasa_res_mean:
        sasa_res_sum = np.zeros((top.n_residues,), dtype=np.float64)
        sasa_res_frames = 0

    global_frame = 0

    print("\n=== ANALYSIS START ===")
    print(f"System          : {system_name}")
    print(f"Replica         : {rep_name}")
    print(f"Topology        : {top_path}")
    print(f"Segments        : {[p.name for p in segs]}")
    print(f"ps/frame        : {ps_per_frame} ps")
    print(f"Align mode      : {args.align}")
    print(f"Distance        : res{args.dist_res1}:{args.dist_atom} -> res{args.dist_res2}:{args.dist_atom}  (atom idx {dist_a}->{dist_b})")
    if extra_pair:
        e1, e2, r1, a1, r2, a2 = extra_pair
        print(f"Extra distance  : res{r1}:{a1} -> res{r2}:{a2}  (atom idx {e1}->{e2})")
    print(f"Total SASA      : {do_sasa_total}")
    print(f"Mean res SASA   : {do_sasa_res_mean}")
    print("Outputs:")
    print(f"  {timeseries_csv}")
    print(f"  {rmsf_csv}")
    if do_sasa_res_mean:
        print(f"  {sasa_res_mean_csv}")

    for seg_i, seg_path in enumerate(segs, start=1):
        print(f"\n--- Segment {seg_i}: {seg_path.name} ---")
        # iterate chunked
        for chunk in md.iterload(str(seg_path), top=str(top_path), chunk=args.chunk_frames):
            # Align
            if align_idx is not None:
                chunk.superpose(ref, 0, atom_indices=align_idx)

            n = chunk.n_frames
            frames = np.arange(global_frame, global_frame + n, dtype=np.int64)
            time_ps = frames.astype(np.float64) * ps_per_frame

            # RMSD to ref using CA for stability (even if align=backbone)
            rmsd_nm = md.rmsd(chunk, ref, atom_indices=ca_idx)

            # Distance GLN(8)-ASN(19) (or whatever indices you pass)
            d_nm = md.compute_distances(chunk, [[dist_a, dist_b]])[:, 0]

            # Radius of gyration (nm) (uses all atoms by default in mdtraj)
            rg_nm = md.compute_rg(chunk)

            # Total SASA per frame
            if do_sasa_total:
                sasa_atom_nm2 = md.shrake_rupley(chunk, mode="atom", n_sphere_points=args.sasa_sphere_points)
                sasa_total_nm2 = sasa_atom_nm2.sum(axis=1)
            else:
                sasa_total_nm2 = None

            # Optional mean per-residue SASA accumulator
            if do_sasa_res_mean:
                sasa_res_nm2 = md.shrake_rupley(chunk, mode="residue", n_sphere_points=args.sasa_sphere_points)
                sasa_res_sum += sasa_res_nm2.sum(axis=0)
                sasa_res_frames += n

            # Optional extra distance
            if extra_pair:
                e1, e2, r1, a1, r2, a2 = extra_pair
                d2_nm = md.compute_distances(chunk, [[e1, e2]])[:, 0]
            else:
                d2_nm = None

            # RMSF accumulators
            xyz_ca = chunk.xyz[:, ca_idx, :]  # nm
            sum_xyz += xyz_ca.sum(axis=0)
            sum_xyz2 += (xyz_ca ** 2).sum(axis=0)
            total_frames += n

            # Build timeseries df (stream write)
            data = {
                "system": system_name,
                "replica": rep_name,
                "segment": seg_path.name,
                "segment_index": seg_i,
                "frame_global": frames,
                "time_ps": time_ps,
                "rmsd_nm": rmsd_nm,
                f"dist_res{args.dist_res1}_{args.dist_atom}_to_res{args.dist_res2}_{args.dist_atom}_nm": d_nm,
                "rg_nm": rg_nm,
            }
            if do_sasa_total:
                data["sasa_total_nm2"] = sasa_total_nm2
            if d2_nm is not None:
                data[f"dist_res{r1}_{a1}_to_res{r2}_{a2}_nm"] = d2_nm

            df_ts = pd.DataFrame(data)
            append_csv(df_ts, timeseries_csv)

            global_frame += n

            # simple progress print
            print(f"  wrote frames {int(frames[0])}..{int(frames[-1])}  (total={global_frame})")

    # Final RMSF (CA) per residue over all frames
    if total_frames == 0:
        raise RuntimeError("No frames processed (total_frames=0).")

    mean_xyz = sum_xyz / float(total_frames)
    mean_xyz2 = sum_xyz2 / float(total_frames)
    var_xyz = mean_xyz2 - (mean_xyz ** 2)
    rmsf_nm = np.sqrt(var_xyz.sum(axis=1))  # nm per CA

    # Map CA -> residues (one row per CA/residue)
    res_map = residue_table_for_ca(top, ca_idx)
    df_rmsf = res_map.copy()
    df_rmsf["system"] = system_name
    df_rmsf["replica"] = rep_name
    df_rmsf["rmsf_ca_nm"] = rmsf_nm.astype(np.float64)
    df_rmsf = df_rmsf[["system", "replica", "res_index", "resSeq", "resname", "rmsf_ca_nm"]].sort_values(["res_index"])
    df_rmsf.to_csv(rmsf_csv, index=False)

    # Mean per-residue SASA over all frames (optional)
    if do_sasa_res_mean:
        if sasa_res_frames == 0:
            raise RuntimeError("Requested sasa_per_res_mean but processed 0 frames for it.")
        sasa_res_mean_nm2 = sasa_res_sum / float(sasa_res_frames)
        # Create residue table
        rows = []
        for res in top.residues:
            resseq = getattr(res, "resSeq", None)
            rows.append(
                {
                    "system": system_name,
                    "replica": rep_name,
                    "res_index": res.index,
                    "resSeq": int(resseq) if resseq is not None else res.index,
                    "resname": res.name,
                    "sasa_mean_nm2": float(sasa_res_mean_nm2[res.index]),
                }
            )
        pd.DataFrame(rows).to_csv(sasa_res_mean_csv, index=False)

    print("\n=== ANALYSIS DONE ===")
    print("Wrote:")
    print(" ", timeseries_csv)
    print(" ", rmsf_csv)
    if do_sasa_res_mean:
        print(" ", sasa_res_mean_csv)


if __name__ == "__main__":
    main()
