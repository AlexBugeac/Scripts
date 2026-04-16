#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def welford_init(n_atoms: int):
    mean = np.zeros((n_atoms, 3), dtype=np.float64)
    m2   = np.zeros((n_atoms, 3), dtype=np.float64)
    count = 0
    return mean, m2, count


def welford_update(mean, m2, count, xyz):  # xyz: (frames, atoms, 3)
    # Online update to avoid storing all frames
    for i in range(xyz.shape[0]):
        count += 1
        delta = xyz[i] - mean
        mean += delta / count
        delta2 = xyz[i] - mean
        m2 += delta * delta2
    return mean, m2, count


def welford_rmsf(mean, m2, count):
    if count < 2:
        return np.zeros((mean.shape[0],), dtype=np.float64)
    var = m2 / (count - 1)     # per-axis variance
    msd = var.sum(axis=1)      # x+y+z
    return np.sqrt(msd)        # RMSF per atom


def main():
    ap = argparse.ArgumentParser(
        description="Generate RMSD (CA), RMSF (CA), SASA (total), and distance(atom i, atom j) from PRMTOP + DCDs."
    )
    ap.add_argument("-t", "--top", required=True, type=Path, help="PRMTOP topology")
    ap.add_argument("-d", "--dcd", required=True, nargs="+", type=Path, help="One or more DCDs (in order)")
    ap.add_argument("-o", "--out-prefix", required=True, help="Output prefix for CSV files")

    ap.add_argument("-n", "--nframes", type=int, default=250000, help="Max frames to process across all DCDs (default 250000)")
    ap.add_argument("--chunk", type=int, default=2000, help="Frames per iterload chunk (default 2000)")
    ap.add_argument("--align-sel", default="name CA", help="Selection for RMSD/RMSF alignment (default: name CA)")

    ap.add_argument("--atom1", type=int, default=9, help="Atom index 1 for distance (0-based, default 9)")
    ap.add_argument("--atom2", type=int, default=20, help="Atom index 2 for distance (0-based, default 20)")

    ap.add_argument("--sasa-stride", type=int, default=1,
                    help="Compute SASA every k frames (default 1 = every frame). "
                         "Use e.g. 10/50 to speed up; frames not computed will be omitted from SASA CSV.")
    ap.add_argument("--no-sasa", action="store_true", help="Skip SASA entirely (fast).")
    args = ap.parse_args()

    top = args.top.expanduser().resolve()
    dcds = [p.expanduser().resolve() for p in args.dcd]
    if not top.exists():
        raise FileNotFoundError(str(top))
    for p in dcds:
        if not p.exists():
            raise FileNotFoundError(str(p))

    import mdtraj as md

    # Reference = first frame of first DCD
    ref = md.load_frame(str(dcds[0]), 0, top=str(top))

    # Selection for CA-based RMSD/RMSF
    sel_atoms = ref.topology.select(args.align_sel)
    if sel_atoms.size == 0:
        raise ValueError(f"Selection '{args.align_sel}' returned 0 atoms.")
    sel_atoms = sel_atoms.astype(int)

    # Validate atom indices exist
    n_atoms_total = ref.n_atoms
    if not (0 <= args.atom1 < n_atoms_total) or not (0 <= args.atom2 < n_atoms_total):
        raise ValueError(f"Atom indices out of range for topology: n_atoms={n_atoms_total}, "
                         f"atom1={args.atom1}, atom2={args.atom2}")

    # Welford accumulators for RMSF on selected atoms
    mean, m2, count = welford_init(len(sel_atoms))

    rmsd_rows = []
    dist_rows = []
    sasa_rows = []

    processed = 0
    stride = max(1, int(args.sasa_stride))

    print(f"[INFO] Topology atoms: {n_atoms_total}")
    print(f"[INFO] Alignment/RMSF selection '{args.align_sel}' -> {len(sel_atoms)} atoms")
    print(f"[INFO] Distance atoms (0-based): {args.atom1}, {args.atom2}")
    print(f"[INFO] Target frames: {args.nframes}")
    if args.no_sasa:
        print("[INFO] SASA: skipped")
    else:
        print(f"[INFO] SASA: total per frame, computed every {stride} frame(s)")

    for dcd in dcds:
        if processed >= args.nframes:
            break

        for traj in md.iterload(str(dcd), top=str(top), chunk=args.chunk):
            if processed >= args.nframes:
                break

            remaining = args.nframes - processed
            if traj.n_frames > remaining:
                traj = traj[:remaining]

            # RMSD (nm), aligned on selected atoms
            rmsd = md.rmsd(traj, ref, atom_indices=sel_atoms)

            # Distance (nm) between two atom indices
            v = traj.xyz[:, args.atom2, :] - traj.xyz[:, args.atom1, :]
            dist = np.linalg.norm(v, axis=1)

            # RMSF update (selected atoms)
            xyz_sel = traj.xyz[:, sel_atoms, :]
            mean, m2, count = welford_update(mean, m2, count, xyz_sel)

            # SASA (total nm^2), optionally strided
            if not args.no_sasa:
                # select frames in this chunk for SASA
                # global frame indices = processed + i
                idx_local = [i for i in range(traj.n_frames) if ((processed + i) % stride == 0)]
                if idx_local:
                    sub = traj[idx_local]
                    sasa_atom = md.shrake_rupley(sub, mode="atom")  # (frames, atoms)
                    sasa_total = sasa_atom.sum(axis=1)              # (frames,)
                else:
                    sasa_total = None
            else:
                idx_local = []
                sasa_total = None

            # Append rows
            for i in range(traj.n_frames):
                gf = processed + i
                rmsd_rows.append({"frame": gf, "rmsd_nm": float(rmsd[i])})
                dist_rows.append({"frame": gf, "dist_nm": float(dist[i])})

            if sasa_total is not None:
                for j, i_local in enumerate(idx_local):
                    gf = processed + i_local
                    sasa_rows.append({"frame": gf, "sasa_nm2": float(sasa_total[j])})

            processed += traj.n_frames

            if processed % 20000 < traj.n_frames:  # crossed a 20k boundary
                print(f"[PROGRESS] processed {processed}/{args.nframes} frames...")

    # Write time-series CSVs
    pref = args.out_prefix
    pd.DataFrame(rmsd_rows).to_csv(f"{pref}_rmsd.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(f"{pref}_dist_a{args.atom1}_a{args.atom2}.csv", index=False)
    if not args.no_sasa:
        pd.DataFrame(sasa_rows).to_csv(f"{pref}_sasa.csv", index=False)

    # RMSF per residue (from selected atoms)
    rmsf_atoms = welford_rmsf(mean, m2, count)

    atoms_list = list(ref.topology.atoms)
    atom_meta = []
    for aidx, rmsf in zip(sel_atoms, rmsf_atoms):
        atom = atoms_list[int(aidx)]
        res = atom.residue
        atom_meta.append((res.chain.index, res.index, res.resSeq, res.name, float(rmsf)))

    df_rmsf = pd.DataFrame(atom_meta, columns=["chain_index", "res_index", "resSeq", "resname", "rmsf_nm"])
    # If selection is CA, there should be 1 per residue; aggregate anyway (robust)
    df_rmsf = (
        df_rmsf.groupby(["chain_index", "res_index", "resSeq", "resname"], as_index=False)["rmsf_nm"]
        .mean()
        .sort_values(["chain_index", "res_index"])
    )
    df_rmsf.to_csv(f"{pref}_rmsf.csv", index=False)

    print(f"[OK] Done. Processed {processed} frames (max {args.nframes}).")
    print(f"[OK] Wrote: {pref}_rmsd.csv")
    print(f"[OK] Wrote: {pref}_dist_a{args.atom1}_a{args.atom2}.csv")
    if not args.no_sasa:
        print(f"[OK] Wrote: {pref}_sasa.csv (stride={stride})")
    print(f"[OK] Wrote: {pref}_rmsf.csv")


if __name__ == "__main__":
    main()
