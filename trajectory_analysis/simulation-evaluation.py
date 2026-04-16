#!/usr/bin/env python3

import argparse
import mdtraj as md
import numpy as np
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser(
        description="Compute RMSD, RMSF, and end-to-end distance from DCD trajectory"
    )
    parser.add_argument("-t", "--topology", required=True, help="Topology file (PRMTOP/PDB)")
    parser.add_argument("-d", "--dcd", required=True, help="Trajectory file (DCD)")
    parser.add_argument("--stride", type=int, default=1, help="Stride for trajectory loading")

    # End-to-end options
    parser.add_argument("--start-res", type=int, required=True,
                        help="Start residue number (resSeq)")
    parser.add_argument("--end-res", type=int, required=True,
                        help="End residue number (resSeq)")
    parser.add_argument("--atom", default="CA",
                        help="Atom name to use for end-to-end distance (default: CA)")

    args = parser.parse_args()

    # ======================
    # Auto-generate output prefix
    # ======================
    system_name = os.path.splitext(os.path.basename(args.topology))[0]
    replica_name = os.path.basename(os.path.dirname(os.path.abspath(args.dcd)))
    out_prefix = f"{system_name}_{replica_name}"

    print(f"System   : {system_name}")
    print(f"Replica  : {replica_name}")
    print(f"Out base : {out_prefix}")

    # ======================
    # Load trajectory
    # ======================
    print("Loading trajectory...")
    traj = md.load(args.dcd, top=args.topology, stride=args.stride)
    top = traj.topology

    # ======================
    # CA selection (global)
    # ======================
    ca_atoms = top.select("name CA")
    if len(ca_atoms) < 2:
        raise ValueError("Not enough CA atoms found.")

    traj_ca = traj.atom_slice(ca_atoms)

    # ======================
    # RMSD (global)
    # ======================
    print("Computing global RMSD...")
    rmsd = md.rmsd(traj_ca, traj_ca, frame=0)

    pd.DataFrame({
        "frame": np.arange(len(rmsd)),
        "rmsd_A": rmsd * 10.0
    }).to_csv(f"{out_prefix}_rmsd.csv", index=False)

    # ======================
    # RMSD residues 1–70 (NEW)
    # ======================
    print("Computing RMSD for residues 1–70...")

    sel_1_70 = top.select("resSeq >= 1 and resSeq <= 70 and name CA")
    if len(sel_1_70) < 2:
        raise ValueError("Not enough CA atoms found for residues 1–70.")

    traj_1_70 = traj.atom_slice(sel_1_70)
    rmsd_1_70 = md.rmsd(traj_1_70, traj_1_70, frame=0)

    pd.DataFrame({
        "frame": np.arange(len(rmsd_1_70)),
        "rmsd_1_70_A": rmsd_1_70 * 10.0
    }).to_csv(f"{out_prefix}_rmsd_1_70.csv", index=False)

    # ======================
    # RMSF (global CA)
    # ======================
    print("Computing RMSF...")
    rmsf = md.rmsf(traj_ca, traj_ca)

    residues = [
        traj_ca.topology.atom(i).residue.resSeq
        for i in range(len(rmsf))
    ]

    pd.DataFrame({
        "residue": residues,
        "rmsf_A": rmsf * 10.0
    }).to_csv(f"{out_prefix}_rmsf.csv", index=False)

    # ======================
    # End-to-end distance
    # ======================
    print("Computing end-to-end distance...")

    start_sel = f"resSeq {args.start_res} and name {args.atom}"
    end_sel   = f"resSeq {args.end_res} and name {args.atom}"

    start_atoms = top.select(start_sel)
    end_atoms   = top.select(end_sel)

    if len(start_atoms) != 1 or len(end_atoms) != 1:
        raise ValueError(
            f"Selection error:\n"
            f"  start: '{start_sel}' → {len(start_atoms)} atoms\n"
            f"  end:   '{end_sel}' → {len(end_atoms)} atoms"
        )

    distances = md.compute_distances(
        traj,
        [[start_atoms[0], end_atoms[0]]]
    ).flatten()

    pd.DataFrame({
        "frame": np.arange(len(distances)),
        "end_to_end_A": distances * 10.0
    }).to_csv(f"{out_prefix}_end_to_end.csv", index=False)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
