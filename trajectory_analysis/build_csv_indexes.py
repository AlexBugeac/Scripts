#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

# -------- CONFIG --------
ROOTS = [
    "/storage1/simulations/openmm",
]

SIM_PREFIXES = [
    "8RJJ-native",
    "8RJJ-m_I411C_S424C_db411-424",
    "8RJJ-m_I411C_S424C_db424-429",
]

REPLICAS = ["replica_01", "replica_02", "replica_03"]
DCD_CANDIDATES = ["traj.dcd", "traj2.dcd", "traj3.dcd"]  # only these

# Keep False for huge trajectories (mdtraj merge loads into RAM)
WRITE_MERGED_DCD = False
# ------------------------


def find_sim_dirs():
    sim_dirs = []
    for root in ROOTS:
        rootp = Path(root)
        if not rootp.exists():
            continue
        for pref in SIM_PREFIXES:
            for p in rootp.rglob(pref):
                if p.is_dir():
                    sim_dirs.append(p)
    return sorted(set(sim_dirs))


def load_time_axis(replica_dir: Path):
    """
    Best-effort load time/step per logged row from log.csv/log2.csv.
    Returns dataframe with columns: log_row, time_ps, step
    """
    for logname in ("log.csv", "log2.csv"):
        lp = replica_dir / logname
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


def count_frames_with_mdtraj(topology_path: Path, dcd_path: Path) -> int:
    """
    Count frames robustly without loading all frames at once.
    """
    import mdtraj as md

    n = 0
    for chunk in md.iterload(str(dcd_path), top=str(topology_path), chunk=5000):
        n += chunk.n_frames
    return n


def build_index_for_replica(sim_dir: Path, replica: str):
    rdir = sim_dir / replica
    if not rdir.exists():
        return None

    dcds = []
    for name in DCD_CANDIDATES:
        p = rdir / name
        if p.exists():
            dcds.append(p)

    if not dcds:
        return None

    prmtops = list(sim_dir.glob("*.prmtop"))
    if not prmtops:
        raise FileNotFoundError(
            f"No PRMTOP found in {sim_dir}. Put a *.prmtop in the sim folder."
        )
    top = prmtops[0]

    frame_counts = [count_frames_with_mdtraj(top, d) for d in dcds]

    rows = []
    global_frame = 0
    for dcd_path, n_frames in zip(dcds, frame_counts):
        for sf in range(n_frames):
            rows.append(
                {
                    "global_frame": global_frame,
                    "source_file": dcd_path.name,
                    "source_frame": sf,
                }
            )
            global_frame += 1

    index_df = pd.DataFrame(rows)

    # Add identifiers inside the CSV (super helpful when concatenating later)
    index_df["simulation"] = sim_dir.name
    index_df["replica"] = replica

    # Attach time axis if possible (best effort)
    log_df = load_time_axis(rdir)
    if log_df is not None:
        minlen = min(len(index_df), len(log_df))
        index_df.loc[: minlen - 1, "time_ps"] = log_df.loc[: minlen - 1, "time_ps"].values
        index_df.loc[: minlen - 1, "step"] = log_df.loc[: minlen - 1, "step"].values
    else:
        index_df["time_ps"] = pd.NA
        index_df["step"] = pd.NA

    return index_df, top, dcds


def maybe_write_merged_dcd(out_dir: Path, top: Path, dcds):
    """
    Optional convenience merged DCD (RAM-heavy). Keep WRITE_MERGED_DCD=False for large runs.
    """
    import mdtraj as md

    merged_path = out_dir / "merged_traj.dcd"
    trajs = [md.load(str(d), top=str(top)) for d in dcds]
    merged = trajs[0].join(trajs[1:]) if len(trajs) > 1 else trajs[0]
    merged.save_dcd(str(merged_path))
    return merged_path


def main():
    sim_dirs = find_sim_dirs()
    if not sim_dirs:
        print("No simulation directories found. Edit ROOTS/SIM_PREFIXES in the script.")
        return

    print(f"Found {len(sim_dirs)} sim dirs:")
    for s in sim_dirs:
        print(" -", s)

    for sim_dir in sim_dirs:
        for rep in REPLICAS:
            try:
                result = build_index_for_replica(sim_dir, rep)
            except Exception as e:
                print(f"[SKIP] {sim_dir}/{rep}: {e}")
                continue

            if result is None:
                print(f"[SKIP] {sim_dir}/{rep}: no traj*.dcd found")
                continue

            index_df, top, dcds = result
            out_dir = sim_dir / rep
            sim_name = sim_dir.name
            rep_name = rep.replace("_", "")  # replica_01 -> replica01
            out_csv = out_dir / f"{sim_name}_{rep_name}_trajectory_index.csv"

            index_df.to_csv(out_csv, index=False)
            print(f"[OK] wrote {out_csv} (frames={len(index_df)}) dcds={[d.name for d in dcds]}")

            if WRITE_MERGED_DCD:
                merged_path = maybe_write_merged_dcd(out_dir, top, dcds)
                print(f"     merged dcd -> {merged_path}")


if __name__ == "__main__":
    main()
