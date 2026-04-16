#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
import mdtraj as md

def pick_first_last_ca(traj: md.Trajectory):
    ca = traj.topology.select("protein and name CA")
    if ca.size < 2:
        raise RuntimeError("Could not find at least 2 Cα atoms (protein and name CA).")
    return int(ca[0]), int(ca[-1])

def residue_table(traj: md.Trajectory):
    rows = []
    for r in traj.topology.residues:
        if r.is_protein:
            resseq = getattr(r, "resSeq", None)
            rows.append({
                "resIndex0": r.index,
                "resSeq": int(resseq) if resseq is not None else r.index + 1,
                "resName": r.name
            })
    return pd.DataFrame(rows)

def dcd_completed_frames(dcd_path: str) -> int:
    """
    Estimate how many FULL frames exist in a DCD even if it's still being written.
    We use:
      - header's n_atoms and n_frames (what the writer *intends*)
      - file size to estimate how many complete frames are physically present
    Then we clamp to the smaller of the two.
    """
    from mdtraj.formats import DCDTrajectoryFile

    size = os.path.getsize(dcd_path)
    with DCDTrajectoryFile(dcd_path, mode="r") as f:
        n_atoms = int(f.n_atoms)
        header_n_frames = int(f.n_frames)

    # Typical DCD frame bytes = 3 coordinate blocks (x,y,z):
    # each block: 4-byte marker + (n_atoms*4 bytes floats) + 4-byte marker
    # => n_atoms*4 + 8 bytes per block => 3*(n_atoms*4 + 8) = 12*n_atoms + 24 bytes per frame
    bytes_per_frame = 12 * n_atoms + 24

    # We don't rely on header size explicitly; we infer "full frames" by floor-dividing
    # the file size (minus a small remainder) by bytes_per_frame.
    # For growing files, the header + metadata tend to be stable, and the remainder
    # is a decent proxy for non-frame bytes.
    remainder = size % bytes_per_frame
    full_frames_by_size = (size - remainder) // bytes_per_frame

    return int(min(header_n_frames, full_frames_by_size))

def safe_load_dcd(dcd: str, top: str, stride: int, max_backoff_frames: int = 5) -> md.Trajectory:
    """
    Load only up to the last COMPLETELY written frame.
    If the file is being written and MDTraj still complains, back off a few frames and retry.
    """
    n_full = dcd_completed_frames(dcd)
    if n_full <= 0:
        raise RuntimeError(f"No complete frames detected in {dcd}")

    # First load all completed frames (no stride), then apply stride.
    # If loading fails, back off a few frames and retry.
    last_err = None
    for backoff in range(0, max_backoff_frames + 1):
        n_try = n_full - backoff
        if n_try <= 0:
            break
        try:
            t_all = md.load_dcd(dcd, top=top, n_frames=n_try)
            return t_all[::stride] if stride > 1 else t_all
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to load a stable subset of frames from {dcd}. Last error: {last_err}")

def load_and_compute(top, dcd, stride, align_sel, dt_ps, end_atom1, end_atom2, n_sphere_points):
    # Safe loading (works even if DCD is still being written, as long as some full frames exist)
    t = safe_load_dcd(dcd=dcd, top=top, stride=stride)

    ref = t[0]

    align_idx = t.topology.select(align_sel)
    if align_idx.size == 0:
        raise RuntimeError(f"Alignment selection returned 0 atoms: {align_sel}")

    # Align trajectory to its first frame (per-traj reference)
    t.superpose(ref, atom_indices=align_idx)

    # RMSD (nm -> Å)
    rmsd_A = md.rmsd(t, ref, atom_indices=align_idx) * 10.0

    # End-to-end atoms (default: first/last protein CA)
    a1, a2 = end_atom1, end_atom2
    if a1 is None or a2 is None:
        a1, a2 = pick_first_last_ca(t)
    end_to_end_A = md.compute_distances(t, [[a1, a2]])[:, 0] * 10.0

    # SASA: atom-mode -> sum per frame (nm^2 -> Å^2)
    sasa_nm2 = md.shrake_rupley(t, mode="atom", n_sphere_points=n_sphere_points)
    sasa_A2 = sasa_nm2.sum(axis=1) * 100.0

    frame = np.arange(t.n_frames)
    time_ps = np.full_like(frame, np.nan, dtype=float) if dt_ps is None else frame * float(dt_ps)

    # RMSF per residue using protein Cα (nm -> Å)
    ca = t.topology.select("protein and name CA")
    if ca.size == 0:
        raise RuntimeError("No protein Cα found for RMSF.")
    rmsf_A = md.rmsf(t, ref, atom_indices=ca) * 10.0
    ca_res_idx = np.array([t.topology.atom(i).residue.index for i in ca], dtype=int)

    return t, a1, a2, frame, time_ps, rmsd_A, end_to_end_A, sasa_A2, ca_res_idx, rmsf_A

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", required=True, help="PRMTOP or PDB topology")
    ap.add_argument("--system", required=True)
    ap.add_argument("--replica", required=True, type=int)
    ap.add_argument("--replica_dir", required=True, help="Path to replica_01/02/03 directory")
    ap.add_argument("--dcd_glob", default="traj*.dcd", help='Which DCDs to include, default "traj*.dcd"')
    ap.add_argument("--outdir", default="csv_out")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dt_ps", type=float, default=None)
    ap.add_argument("--align_sel", default="protein and name CA")
    ap.add_argument("--end_atom1", type=int, default=None)
    ap.add_argument("--end_atom2", type=int, default=None)
    ap.add_argument("--n_sphere_points", type=int, default=240, help="SASA resolution; 240 fast, 960 higher-res")
    ap.add_argument("--skip_on_error", action="store_true", help="If set, skip a traj if it cannot be read safely")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    dcdfiles = sorted(glob.glob(os.path.join(args.replica_dir, args.dcd_glob)))
    if not dcdfiles:
        raise RuntimeError(f"No DCDs matched: {os.path.join(args.replica_dir, args.dcd_glob)}")

    master_ts_path   = os.path.join(args.outdir, f"{args.system}_rep{args.replica}_timeseries_ALL.csv")
    master_rmsf_path = os.path.join(args.outdir, f"{args.system}_rep{args.replica}_rmsf_ALL.csv")

    ts_rows = []
    rmsf_rows = []

    for dcd in dcdfiles:
        traj_label = os.path.basename(dcd)

        print(f"\n=== Processing {traj_label} (stride={args.stride}) ===")
        try:
            t, a1, a2, frame, time_ps, rmsd_A, end_to_end_A, sasa_A2, ca_res_idx, rmsf_A = load_and_compute(
                top=args.top,
                dcd=dcd,
                stride=args.stride,
                align_sel=args.align_sel,
                dt_ps=args.dt_ps,
                end_atom1=args.end_atom1,
                end_atom2=args.end_atom2,
                n_sphere_points=args.n_sphere_points
            )
        except Exception as e:
            msg = f"ERROR on {traj_label}: {e}"
            if args.skip_on_error:
                print(msg)
                print("Skipping this trajectory and continuing.")
                continue
            raise

        # Timeseries rows
        df_ts = pd.DataFrame({
            "system": args.system,
            "replica": args.replica,
            "traj": traj_label,
            "frame": frame,
            "time_ps": time_ps,
            "rmsd_A": rmsd_A,
            "end_to_end_A": end_to_end_A,
            "sasa_A2": sasa_A2,
        })
        ts_rows.append(df_ts)

        # RMSF rows (merge residue info)
        res_df = residue_table(t)
        df_rmsf = pd.DataFrame({
            "system": args.system,
            "replica": args.replica,
            "traj": traj_label,
            "resIndex0": ca_res_idx,
            "rmsf_A": rmsf_A
        }).merge(res_df, on="resIndex0", how="left") \
          .drop(columns=["resIndex0"]) \
          .sort_values("resSeq")
        rmsf_rows.append(df_rmsf)

        print(f"Frames loaded: {t.n_frames} | End-to-end atoms (0-based): {a1}, {a2}")

    if not ts_rows:
        raise RuntimeError("No trajectories were successfully processed (ts_rows empty).")

    # Write combined outputs
    df_ts_all = pd.concat(ts_rows, ignore_index=True)
    df_rmsf_all = pd.concat(rmsf_rows, ignore_index=True) if rmsf_rows else pd.DataFrame()

    df_ts_all.to_csv(master_ts_path, index=False)
    if not df_rmsf_all.empty:
        df_rmsf_all.to_csv(master_rmsf_path, index=False)

    print("\nWrote combined CSVs:")
    print(" ", master_ts_path)
    if not df_rmsf_all.empty:
        print(" ", master_rmsf_path)

if __name__ == "__main__":
    main()
