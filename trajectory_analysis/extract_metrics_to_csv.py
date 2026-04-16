#!/usr/bin/env python3
import os
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", required=True, help="PRMTOP or PDB topology")
    ap.add_argument("--dcd", required=True, help="Trajectory DCD")
    ap.add_argument("--system", required=True, help="System name label for CSV")
    ap.add_argument("--replica", required=True, type=int)
    ap.add_argument("--traj_name", default=None)
    ap.add_argument("--outdir", default="csv_out")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dt_ps", type=float, default=None)
    ap.add_argument("--align_sel", default="protein and name CA")
    ap.add_argument("--end_atom1", type=int, default=None)
    ap.add_argument("--end_atom2", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    traj_label = args.traj_name or os.path.basename(args.dcd)

    t = md.load_dcd(args.dcd, top=args.top, stride=args.stride)

    ref = t[0]

    align_idx = t.topology.select(args.align_sel)
    if align_idx.size == 0:
        raise RuntimeError(f"Alignment selection returned 0 atoms: {args.align_sel}")

    t.superpose(ref, atom_indices=align_idx)

    rmsd_A = md.rmsd(t, ref, atom_indices=align_idx) * 10.0

    a1 = args.end_atom1
    a2 = args.end_atom2
    if a1 is None or a2 is None:
        a1, a2 = pick_first_last_ca(t)

    end_to_end_A = md.compute_distances(t, [[a1, a2]])[:, 0] * 10.0
    
    sasa_nm2 = md.shrake_rupley(t, mode="atom")
    sasa_A2 = sasa_nm2.sum(axis=1)*100

    frame = np.arange(t.n_frames)
    if args.dt_ps is None:
        time_ps = np.full_like(frame, np.nan, dtype=float)
    else:
        time_ps = frame * float(args.dt_ps)

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

    out_ts = os.path.join(args.outdir,
        f"{args.system}_rep{args.replica}_{traj_label}_timeseries.csv")
    df_ts.to_csv(out_ts, index=False)

    ca = t.topology.select("protein and name CA")
    if ca.size == 0:
        raise RuntimeError("No protein Cα found for RMSF.")

    rmsf_A = md.rmsf(t, ref, atom_indices=ca) * 10.0

    res_df = residue_table(t)
    ca_res_idx = np.array(
        [t.topology.atom(i).residue.index for i in ca],
        dtype=int
    )

    df_rmsf = pd.DataFrame({
        "system": args.system,
        "replica": args.replica,
        "traj": traj_label,
        "resIndex0": ca_res_idx,
        "rmsf_A": rmsf_A
    }).merge(res_df, on="resIndex0", how="left") \
      .drop(columns=["resIndex0"]) \
      .sort_values("resSeq")

    out_rmsf = os.path.join(args.outdir,
        f"{args.system}_rep{args.replica}_{traj_label}_rmsf.csv")
    df_rmsf.to_csv(out_rmsf, index=False)

    print("Wrote:")
    print(" ", out_ts)
    print(" ", out_rmsf)
    print(f"End-to-end atoms (0-based): {a1} {a2}")

if __name__ == "__main__":
    main()
