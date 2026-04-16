#!/usr/bin/env python3
"""
Replicate-aware stability analysis for E2 (native vs mutants), using MDTraj.

What this script does (per GROUP, e.g., native / mut1 / mut2):
  1) Loads each replicate trajectory separately (rep1, rep2, rep3, ...).
  2) Aligns each replicate to a single reference PDB (CA atoms).
  3) Computes metrics per replicate:
       - Global RMSD (CA) vs reference
       - Global radius of gyration (heavy atoms)
       - Native contacts Q(t) (CA-native contacts from reference)
       - ROI RMSD (CA in ROI) vs reference (if --roi)
       - ROI SASA total (if --sasa)
       - Disulfide S–S distance + chi3 (if --disulfide)
       - Water near ROI (count) (if --water_shell_A and solvent present)
       - DSSP fractions (if --compute_dssp and DSSP available)
  4) Aggregates replicates:
       - Mean ± SD, Mean ± SEM curves (default: truncate to shortest replicate length)
       - Writes aggregate CSVs and plots

Inputs:
  --reference REF.pdb
  --group NAME:TOPO:REP1[,REP2,...]   (repeat --group for each condition)
Optional:
  --roi "412-430,460-490"
  --domains core=300-480 front=481-550 back=551-650  (repeatable items)
  --disulfide "411-424" --disulfide "424-429"
  --sasa
  --compute_dssp
  --water_shell_A 5.0
  --stride 10
  --aggregate_mode truncate|resample  (truncate is safest; resample uses common time grid)

Example:
  python e2_stability_compare_reps.py \
    --out out_e2 \
    --reference native_ref.pdb \
    --group native:native.prmtop:/path/native/replica_01/traj_stride15_1.dcd,/path/native/replica_02/traj_stride15_2.dcd,/path/native/replica_03/traj_stride15_3.dcd \
    --group db411_424:db411_424.prmtop:/path/db411-424/replica_01/traj_stride15_r1.dcd,/path/db411-424/replica_02/traj_stride15_r2.dcd,/path/db411-424/replica_03/traj_stride15_r3.dcd \
    --group db424_429:db424_429.prmtop:/path/db424-429/replica_01/traj_stride15_r1.dcd,/path/db424-429/replica_02/traj_stride15_r2.dcd,/path/db424-429/replica_03/traj_stride15_r3.dcd \
    --roi 405-440 \
    --sasa \
    --disulfide 411-424 \
    --disulfide 424-429 \
    --water_shell_A 5.0 \
    --stride 1

Notes / gotchas:
  - ROI/domain ranges are interpreted as PDB residue numbers (resSeq) IF present in topology.
    Otherwise, they’re treated as 0-based residue indices.
  - Native contacts Q(t) assumes the reference and replicate topologies have the same residue ordering.
    If mutants have missing residues / different atom counts, Q and ROI slices may need mapping by sequence.
"""

import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mdtraj as md


# -------------------------
# Helpers: filesystem
# -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


# -------------------------
# Parsing CLI
# -------------------------

def parse_ranges(ranges_str: str) -> List[Tuple[int, int]]:
    """Parse '412-430,460-490' into inclusive ranges [(412,430),(460,490)]."""
    ranges_str = (ranges_str or "").strip()
    if not ranges_str:
        return []
    out = []
    for chunk in ranges_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", chunk)
        if not m:
            raise ValueError(f"Bad range '{chunk}' (expected 'start-end')")
        a, b = int(m.group(1)), int(m.group(2))
        if b < a:
            a, b = b, a
        out.append((a, b))
    return out

def parse_domains(domains_list: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """Parse domains like ['core=300-480', 'front=481-550,560-570']."""
    domains = {}
    for item in domains_list or []:
        if "=" not in item:
            raise ValueError(f"Bad --domains entry '{item}'. Expected name=ranges")
        name, ranges = item.split("=", 1)
        domains[name.strip()] = parse_ranges(ranges)
    return domains

def parse_pair(pair_str: str) -> Tuple[int, int]:
    """Parse '411-424' into (411,424)."""
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", pair_str)
    if not m:
        raise ValueError(f"Bad pair '{pair_str}' (expected 'a-b')")
    return int(m.group(1)), int(m.group(2))

def parse_group(s: str) -> Tuple[str, str, List[str]]:
    """
    GROUP spec format: NAME:TOPO:REP1[,REP2,...]
    Example: native:native.prmtop:replica_01/traj.dcd,replica_02/traj.dcd,replica_03/traj.dcd
    """
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad --group '{s}'. Expected NAME:TOPO:REP1[,REP2,...]")
    name, topo, reps_str = parts
    reps = [x.strip() for x in reps_str.split(",") if x.strip()]
    if not reps:
        raise ValueError(f"No replicate trajectories provided in group '{name}'")
    return name.strip(), topo.strip(), reps


# -------------------------
# Topology / selection utilities
# -------------------------

def get_residue_id_list(top: md.Topology) -> List[int]:
    """
    Return residue IDs as PDB resSeq if present for all residues, else fallback to 0..n_res-1.
    """
    ids = []
    ok = True
    for r in top.residues:
        rs = getattr(r, "resSeq", None)
        if rs is None:
            ok = False
            break
        ids.append(int(rs))
    if ok and len(ids) == top.n_residues:
        return ids
    return list(range(top.n_residues))

def residue_id_to_index_map(top: md.Topology) -> Dict[int, int]:
    ids = get_residue_id_list(top)
    return {rid: i for i, rid in enumerate(ids)}

def indices_from_ranges(top: md.Topology, ranges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Convert residue-id ranges -> residue indices in this topology.
    """
    if not ranges:
        return np.array([], dtype=int)
    rid2idx = residue_id_to_index_map(top)
    idxs = []
    for a, b in ranges:
        for rid in range(a, b + 1):
            if rid in rid2idx:
                idxs.append(rid2idx[rid])
    return np.array(sorted(set(idxs)), dtype=int)

def atom_indices_for_residues(top: md.Topology, residue_indices: np.ndarray,
                             atom_name_filter: Optional[str] = None) -> np.ndarray:
    residue_set = set(residue_indices.tolist())
    atoms = []
    for atom in top.atoms:
        if atom.residue.index in residue_set:
            if atom_name_filter is None or atom.name == atom_name_filter:
                atoms.append(atom.index)
    return np.array(atoms, dtype=int)

def get_ca_indices(top: md.Topology) -> np.ndarray:
    return np.array([a.index for a in top.atoms if a.name == "CA"], dtype=int)

def get_heavy_atom_indices(top: md.Topology) -> np.ndarray:
    heavy = []
    for a in top.atoms:
        el = getattr(a.element, "symbol", None)
        if el is None:
            if not a.name.upper().startswith("H"):
                heavy.append(a.index)
        else:
            if el != "H":
                heavy.append(a.index)
    return np.array(heavy, dtype=int)

def find_atom_in_residue(top: md.Topology, residue_index: int, atom_name: str) -> Optional[int]:
    res = list(top.residues)[residue_index]
    for a in res.atoms:
        if a.name == atom_name:
            return a.index
    return None


# -------------------------
# Time + alignment
# -------------------------

def time_ns(traj: md.Trajectory) -> np.ndarray:
    if traj.time is None or len(traj.time) != traj.n_frames:
        return np.arange(traj.n_frames, dtype=float)
    return traj.time / 1000.0  # ps -> ns

def align_to_reference(traj: md.Trajectory, ref: md.Trajectory, ca_indices: np.ndarray) -> md.Trajectory:
    return traj.superpose(ref, atom_indices=ca_indices)


# -------------------------
# Metrics
# -------------------------

def compute_rmsd_ca(traj: md.Trajectory, ref: md.Trajectory, ca_idx: np.ndarray) -> np.ndarray:
    return md.rmsd(traj, ref, atom_indices=ca_idx)

def compute_rg_heavy(traj: md.Trajectory) -> np.ndarray:
    heavy = get_heavy_atom_indices(traj.topology)
    return md.compute_rg(traj.atom_slice(heavy))

def compute_native_contacts_Q(traj: md.Trajectory, ref: md.Trajectory, ca_idx: np.ndarray,
                              cutoff_nm: float = 0.8) -> np.ndarray:
    """
    Q(t): fraction of CA-based native contacts present.
    Native contacts defined from reference as CA pairs within cutoff, excluding |i-j| < 3.
    """
    n = len(ca_idx)
    pairs = []
    for i in range(n):
        for j in range(i + 3, n):
            pairs.append((ca_idx[i], ca_idx[j]))
    pairs = np.array(pairs, dtype=int)

    d_ref = md.compute_distances(ref, pairs)[0]
    native_mask = d_ref < cutoff_nm
    native_pairs = pairs[native_mask]
    if native_pairs.shape[0] == 0:
        raise RuntimeError("No native CA contacts found; increase cutoff or check reference.")
    d = md.compute_distances(traj, native_pairs)
    return (d < cutoff_nm).mean(axis=1)

def compute_sasa_per_residue(traj: md.Trajectory) -> np.ndarray:
    return md.shrake_rupley(traj, mode="residue")  # (frames, residues) nm^2

def disulfide_metrics(traj: md.Trajectory, resA_idx: int, resB_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    S–S distance (nm) and chi3 dihedral (radians) for Cys residues with atoms CB and SG.
    """
    top = traj.topology
    sgA = find_atom_in_residue(top, resA_idx, "SG")
    sgB = find_atom_in_residue(top, resB_idx, "SG")
    cbA = find_atom_in_residue(top, resA_idx, "CB")
    cbB = find_atom_in_residue(top, resB_idx, "CB")
    if None in (sgA, sgB, cbA, cbB):
        raise RuntimeError(f"Missing CB/SG for residues {resA_idx} and/or {resB_idx}")
    dist = md.compute_distances(traj, np.array([[sgA, sgB]], dtype=int))[:, 0]
    chi3 = md.compute_dihedrals(traj, np.array([[cbA, sgA, sgB, cbB]], dtype=int))[:, 0]
    return dist, chi3

def water_count_near_atoms(traj: md.Trajectory, roi_atom_indices: np.ndarray, shell_nm: float) -> np.ndarray:
    """
    Count water oxygen atoms within shell_nm of ANY ROI atom per frame.
    Water O heuristic: atom.name == 'O' and residue name in {HOH,WAT,TIP3,SOL}.
    """
    top = traj.topology
    water_O = []
    for a in top.atoms:
        if a.name == "O":
            rn = a.residue.name.upper()
            if rn in {"HOH", "WAT", "TIP3", "SOL"}:
                water_O.append(a.index)
    water_O = np.array(water_O, dtype=int)
    if water_O.size == 0:
        raise RuntimeError("No water oxygens detected (no solvent or different naming).")
    if roi_atom_indices.size == 0:
        raise RuntimeError("ROI atom list is empty; cannot compute water near ROI.")

    # Compute distances for all waterO x ROI atoms
    pairs = np.array([(w, r) for w in water_O for r in roi_atom_indices], dtype=int)
    d = md.compute_distances(traj, pairs)  # (frames, n_pairs)
    d3 = d.reshape(traj.n_frames, water_O.size, roi_atom_indices.size)
    within = (d3 < shell_nm).any(axis=2)  # (frames, n_water)
    return within.sum(axis=1)

def compute_dssp_fractions(traj: md.Trajectory) -> pd.DataFrame:
    dssp = md.compute_dssp(traj, simplified=False)  # (frames, residues)
    helix = np.isin(dssp, ["H", "G", "I"])
    strand = np.isin(dssp, ["E", "B"])
    coil = ~(helix | strand)
    return pd.DataFrame({
        "frac_helix": helix.mean(axis=1),
        "frac_strand": strand.mean(axis=1),
        "frac_coil": coil.mean(axis=1),
    })


# -------------------------
# Aggregation across replicates
# -------------------------

def aggregate_replicates(
    t_list: List[np.ndarray],
    y_list: List[np.ndarray],
    mode: str = "truncate",
    resample_dt: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: t_common, mean, sd, sem

    mode="truncate": truncate all replicates to shortest length (frame-aligned)
    mode="resample": resample each replicate onto a common time grid (linear interp)
    """
    if len(y_list) == 0:
        raise ValueError("No replicates provided for aggregation.")

    if mode == "truncate":
        L = min(len(y) for y in y_list)
        t_common = t_list[0][:L]
        Y = np.vstack([y[:L] for y in y_list])
    elif mode == "resample":
        # Determine common time grid
        t_start = max(t[0] for t in t_list)
        t_end = min(t[-1] for t in t_list)
        if t_end <= t_start:
            raise RuntimeError("Replicates have no overlapping time window for resampling.")
        if resample_dt is None:
            # Guess dt from first replicate median spacing
            dt_guess = np.median(np.diff(t_list[0])) if len(t_list[0]) > 1 else 1.0
            resample_dt = float(dt_guess) if dt_guess > 0 else 1.0
        t_common = np.arange(t_start, t_end + 1e-9, resample_dt)
        Y = []
        for t, y in zip(t_list, y_list):
            Y.append(np.interp(t_common, t, y))
        Y = np.vstack(Y)
    else:
        raise ValueError(f"Unknown aggregate mode: {mode}")

    mean = Y.mean(axis=0)
    sd = Y.std(axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean)
    sem = sd / np.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros_like(mean)
    return t_common, mean, sd, sem


# -------------------------
# Plotting
# -------------------------

def plot_replicates(
    outpath: str,
    title: str,
    xlabel: str,
    ylabel: str,
    t_list: List[np.ndarray],
    y_list: List[np.ndarray],
    labels: List[str]
):
    plt.figure()
    for t, y, lab in zip(t_list, y_list, labels):
        plt.plot(t, y, label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_mean_band(
    outpath: str,
    title: str,
    xlabel: str,
    ylabel: str,
    t: np.ndarray,
    mean: np.ndarray,
    band: np.ndarray,
    band_label: str = "±SD"
):
    plt.figure()
    plt.plot(t, mean, label="mean")
    plt.fill_between(t, mean - band, mean + band, alpha=0.2, label=band_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_hist(
    outpath: str,
    title: str,
    xlabel: str,
    data_dict: Dict[str, np.ndarray],
    bins: int = 60
):
    plt.figure()
    for lab, y in data_dict.items():
        plt.hist(y, bins=bins, alpha=0.5, density=True, label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -------------------------
# Data structures
# -------------------------

@dataclass
class ReplicateResult:
    name: str
    traj: md.Trajectory
    t: np.ndarray
    metrics: Dict[str, np.ndarray]       # time-series metrics
    scalars: Dict[str, float]            # scalar summaries
    extra_tables: Dict[str, pd.DataFrame]  # tables (e.g., SASA per-res avg)


# -------------------------
# Main execution per group
# -------------------------

def analyze_replicate(
    rep_name: str,
    topo_path: str,
    traj_path: str,
    ref: md.Trajectory,
    args
) -> ReplicateResult:
    tr = md.load(traj_path, top=topo_path, stride=args.stride)
    top = tr.topology

    ca = get_ca_indices(top)
    if ca.size == 0:
        raise RuntimeError(f"No CA atoms in topology for replicate {rep_name}")

    # Align to reference (CA)
    tr = align_to_reference(tr, ref, ca_indices=ca)
    t = time_ns(tr)

    metrics: Dict[str, np.ndarray] = {}
    scalars: Dict[str, float] = {}
    tables: Dict[str, pd.DataFrame] = {}

    # Global metrics
    rmsd = compute_rmsd_ca(tr, ref, ca_idx=ca)
    rg = compute_rg_heavy(tr)

    metrics["global_rmsd_nm"] = rmsd
    metrics["global_rg_nm"] = rg

    scalars["global_rmsd_mean_nm"] = float(np.mean(rmsd))
    scalars["global_rmsd_std_nm"] = float(np.std(rmsd))
    scalars["global_rg_mean_nm"] = float(np.mean(rg))
    scalars["global_rg_std_nm"] = float(np.std(rg))

    # Native contacts Q(t)
    if args.compute_Q:
        try:
            Q = compute_native_contacts_Q(tr, ref, ca_idx=ca, cutoff_nm=args.native_contacts_cutoff_nm)
            metrics["native_contacts_Q"] = Q
            scalars["Q_mean"] = float(np.mean(Q))
            scalars["Q_std"] = float(np.std(Q))
        except Exception as e:
            print(f"[WARN] Q(t) failed for {rep_name}: {e}")

    # ROI metrics
    if args.roi_ranges:
        roi_res_idx = indices_from_ranges(top, args.roi_ranges)
        if roi_res_idx.size == 0:
            print(f"[WARN] ROI selection empty for {rep_name}. Check residue numbering.")
        else:
            roi_ca = atom_indices_for_residues(top, roi_res_idx, atom_name_filter="CA")
            if roi_ca.size > 0:
                # Slice traj+ref to those atoms (use local indexing for RMSD)
                tr_roi = tr.atom_slice(roi_ca)
                ref_roi = ref.atom_slice(roi_ca)
                roi_rmsd = md.rmsd(tr_roi, ref_roi, atom_indices=np.arange(tr_roi.n_atoms))
                metrics["roi_rmsd_nm"] = roi_rmsd
                scalars["roi_rmsd_mean_nm"] = float(np.mean(roi_rmsd))
                scalars["roi_rmsd_std_nm"] = float(np.std(roi_rmsd))
            else:
                print(f"[WARN] ROI CA atoms empty for {rep_name}; skipping ROI RMSD.")

            if args.sasa:
                try:
                    sasa = compute_sasa_per_residue(tr)  # (frames, n_res)
                    roi_sasa = sasa[:, roi_res_idx]
                    metrics["roi_sasa_total_nm2"] = roi_sasa.sum(axis=1)
                    scalars["roi_sasa_mean_nm2"] = float(np.mean(metrics["roi_sasa_total_nm2"]))
                    scalars["roi_sasa_std_nm2"] = float(np.std(metrics["roi_sasa_total_nm2"]))

                    # save per-residue average SASA table (whole protein)
                    avg = sasa.mean(axis=0)
                    res_ids = get_residue_id_list(top)
                    df = pd.DataFrame({"residue_id": res_ids, "avg_sasa_nm2": avg})
                    tables["avg_sasa_per_residue"] = df
                except Exception as e:
                    print(f"[WARN] SASA failed for {rep_name}: {e}")

            if args.water_shell_A > 0:
                try:
                    roi_atoms_all = atom_indices_for_residues(top, roi_res_idx, atom_name_filter=None)
                    shell_nm = args.water_shell_A / 10.0
                    wc = water_count_near_atoms(tr, roi_atoms_all, shell_nm=shell_nm)
                    metrics["roi_water_count"] = wc
                    scalars["roi_water_mean"] = float(np.mean(wc))
                    scalars["roi_water_std"] = float(np.std(wc))
                except Exception as e:
                    print(f"[WARN] Water-near-ROI failed for {rep_name}: {e}")

    # DSSP
    if args.compute_dssp:
        try:
            dssp_df = compute_dssp_fractions(tr)
            # store as separate metrics series
            metrics["dssp_frac_helix"] = dssp_df["frac_helix"].to_numpy()
            metrics["dssp_frac_strand"] = dssp_df["frac_strand"].to_numpy()
            metrics["dssp_frac_coil"] = dssp_df["frac_coil"].to_numpy()
            scalars["dssp_helix_mean"] = float(dssp_df["frac_helix"].mean())
            scalars["dssp_strand_mean"] = float(dssp_df["frac_strand"].mean())
            scalars["dssp_coil_mean"] = float(dssp_df["frac_coil"].mean())
        except Exception as e:
            print(f"[WARN] DSSP failed for {rep_name}: {e}")

    # Disulfides
    if args.disulfide_pairs:
        rid2idx = residue_id_to_index_map(top)
        for a_id, b_id in args.disulfide_pairs:
            if a_id not in rid2idx or b_id not in rid2idx:
                print(f"[WARN] Disulfide residues {a_id}-{b_id} not found in {rep_name}")
                continue
            a_idx, b_idx = rid2idx[a_id], rid2idx[b_id]
            try:
                dist, chi3 = disulfide_metrics(tr, a_idx, b_idx)
                key = f"disulfide_{a_id}-{b_id}"
                metrics[f"{key}_dist_nm"] = dist
                metrics[f"{key}_chi3_deg"] = np.degrees(chi3)
                scalars[f"{key}_dist_mean_nm"] = float(np.mean(dist))
                scalars[f"{key}_dist_std_nm"] = float(np.std(dist))
            except Exception as e:
                print(f"[WARN] Disulfide metrics failed for {rep_name} {a_id}-{b_id}: {e}")

    # Domain RMSD (optional)
    if args.domains:
        for dom_name, dom_ranges in args.domains.items():
            dom_res_idx = indices_from_ranges(top, dom_ranges)
            if dom_res_idx.size == 0:
                continue
            dom_ca = atom_indices_for_residues(top, dom_res_idx, atom_name_filter="CA")
            if dom_ca.size == 0:
                continue
            tr_dom = tr.atom_slice(dom_ca)
            ref_dom = ref.atom_slice(dom_ca)
            dom_rmsd = md.rmsd(tr_dom, ref_dom, atom_indices=np.arange(tr_dom.n_atoms))
            metrics[f"domain_{dom_name}_rmsd_nm"] = dom_rmsd
            scalars[f"domain_{dom_name}_rmsd_mean_nm"] = float(np.mean(dom_rmsd))
            scalars[f"domain_{dom_name}_rmsd_std_nm"] = float(np.std(dom_rmsd))

    return ReplicateResult(rep_name, tr, t, metrics, scalars, tables)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--reference", required=True, help="Reference PDB for alignment and native contacts")
    ap.add_argument("--group", action="append", required=True,
                    help="Group spec: NAME:TOPO:REP1[,REP2,...]. Repeat for multiple groups.")
    ap.add_argument("--stride", type=int, default=1, help="Load every Nth frame")
    ap.add_argument("--roi", default="", help="ROI residue ranges, e.g. 412-430,460-490")
    ap.add_argument("--domains", nargs="*", default=[], help="Optional domain ranges: name=ranges")
    ap.add_argument("--sasa", action="store_true", help="Compute SASA (Shrake-Rupley)")
    ap.add_argument("--compute_dssp", action="store_true", help="Compute DSSP fractions (requires DSSP installed)")
    ap.add_argument("--disulfide", action="append", default=[], help="Disulfide pair as residue IDs, e.g. 411-424")
    ap.add_argument("--water_shell_A", type=float, default=0.0,
                    help="If >0 and ROI specified, count waters within this many Å of ROI atoms")
    ap.add_argument("--native_contacts_cutoff_nm", type=float, default=0.8, help="Cutoff for native contacts Q(t)")
    ap.add_argument("--no_Q", action="store_true", help="Disable native contacts Q(t)")
    ap.add_argument("--aggregate_mode", choices=["truncate", "resample"], default="truncate",
                    help="How to aggregate replicates onto a common time axis")
    ap.add_argument("--resample_dt_ns", type=float, default=None,
                    help="If aggregate_mode=resample, set dt in ns (default: inferred)")
    args = ap.parse_args()

    ensure_dir(args.out)

    # Parse ROI / domains / disulfides
    args.roi_ranges = parse_ranges(args.roi) if args.roi else []
    args.domains = parse_domains(args.domains)
    args.disulfide_pairs = [parse_pair(p) for p in (args.disulfide or [])]
    args.compute_Q = not args.no_Q

    # Load reference
    ref = md.load(args.reference)
    ref_ca = get_ca_indices(ref.topology)
    if ref_ca.size == 0:
        raise RuntimeError("Reference has no CA atoms.")
    # Keep ref as single frame (it is), but ensure it’s present
    ref = ref[0]

    # Analyze each group
    all_summary_rows = []

    for gspec in args.group:
        gname, topo, rep_paths = parse_group(gspec)
        gname_s = safe_name(gname)

        group_dir = os.path.join(args.out, gname_s)
        reps_dir = os.path.join(group_dir, "replicates")
        agg_dir = os.path.join(group_dir, "aggregate")
        ensure_dir(reps_dir)
        ensure_dir(agg_dir)

        print(f"\n=== Group: {gname} ===")
        print(f"Topology: {topo}")
        print(f"Replicates: {len(rep_paths)}")

        rep_results: List[ReplicateResult] = []

        # Analyze replicates
        for i, rp in enumerate(rep_paths, start=1):
            rep_name = f"{gname}_rep{i}"
            rep_out = os.path.join(reps_dir, f"rep{i}")
            ensure_dir(rep_out)

            print(f"  -> analyzing {rep_name}: {rp}")
            rr = analyze_replicate(rep_name, topo, rp, ref, args)
            rep_results.append(rr)

            # Save per-replicate CSV of time-series metrics
            df = pd.DataFrame({"time_ns": rr.t})
            for k, v in rr.metrics.items():
                if len(v) == len(rr.t):
                    df[k] = v
            df.to_csv(os.path.join(rep_out, "timeseries_metrics.csv"), index=False)

            # Save per-replicate scalar summary
            pd.DataFrame([rr.scalars]).to_csv(os.path.join(rep_out, "summary_scalars.csv"), index=False)

            # Save extra tables (e.g., avg SASA per residue)
            for tname, tdf in rr.extra_tables.items():
                tdf.to_csv(os.path.join(rep_out, f"{tname}.csv"), index=False)

        # Identify which metrics exist across replicates (intersection)
        metric_keys = set(rep_results[0].metrics.keys())
        for rr in rep_results[1:]:
            metric_keys &= set(rr.metrics.keys())
        metric_keys = sorted(metric_keys)

        # For each metric: plot replicates + aggregate mean±SD/SEM + save CSVs
        for mkey in metric_keys:
            t_list = [rr.t for rr in rep_results]
            y_list = [rr.metrics[mkey] for rr in rep_results]
            labels = [rr.name for rr in rep_results]

            # Replicate overlay plot
            plot_replicates(
                outpath=os.path.join(agg_dir, f"{mkey}_replicates.png"),
                title=f"{gname}: {mkey} (replicates)",
                xlabel="Time (ns)",
                ylabel=mkey,
                t_list=t_list,
                y_list=y_list,
                labels=labels
            )

            # Histograms (helpful for “overall distribution”)
            plot_hist(
                outpath=os.path.join(agg_dir, f"{mkey}_hist.png"),
                title=f"{gname}: {mkey} distribution",
                xlabel=mkey,
                data_dict={lab: y for lab, y in zip(labels, y_list)}
            )

            # Aggregate
            try:
                t_common, mean, sd, sem = aggregate_replicates(
                    t_list, y_list, mode=args.aggregate_mode, resample_dt=args.resample_dt_ns
                )
            except Exception as e:
                print(f"[WARN] Aggregation failed for {gname} metric {mkey}: {e}")
                continue

            # Plot mean±SD
            plot_mean_band(
                outpath=os.path.join(agg_dir, f"{mkey}_mean_sd.png"),
                title=f"{gname}: {mkey} mean ± SD (n={len(rep_results)})",
                xlabel="Time (ns)",
                ylabel=mkey,
                t=t_common,
                mean=mean,
                band=sd,
                band_label="±SD"
            )
            # Plot mean±SEM
            plot_mean_band(
                outpath=os.path.join(agg_dir, f"{mkey}_mean_sem.png"),
                title=f"{gname}: {mkey} mean ± SEM (n={len(rep_results)})",
                xlabel="Time (ns)",
                ylabel=mkey,
                t=t_common,
                mean=mean,
                band=sem,
                band_label="±SEM"
            )

            # Save aggregate CSV
            out_df = pd.DataFrame({
                "time_ns": t_common,
                "mean": mean,
                "sd": sd,
                "sem": sem
            })
            out_df.to_csv(os.path.join(agg_dir, f"{mkey}_aggregate.csv"), index=False)

        # Group-level scalar summary (replicates table + aggregated means)
        rep_scalar_rows = []
        for i, rr in enumerate(rep_results, start=1):
            row = {"group": gname, "replicate": i}
            row.update(rr.scalars)
            rep_scalar_rows.append(row)
        rep_scalar_df = pd.DataFrame(rep_scalar_rows)
        rep_scalar_df.to_csv(os.path.join(agg_dir, "replicate_scalar_summaries.csv"), index=False)

        # Add to global summary table (one row per group with mean±sd across replicates for key scalars)
        # We'll aggregate scalars that exist in all reps.
        common_scalar_keys = set(rep_results[0].scalars.keys())
        for rr in rep_results[1:]:
            common_scalar_keys &= set(rr.scalars.keys())
        common_scalar_keys = sorted(common_scalar_keys)

        group_row = {"group": gname, "n_replicates": len(rep_results)}
        for sk in common_scalar_keys:
            vals = np.array([rr.scalars[sk] for rr in rep_results], dtype=float)
            group_row[f"{sk}_mean_over_reps"] = float(vals.mean())
            group_row[f"{sk}_sd_over_reps"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        all_summary_rows.append(group_row)

    # Write global summary across groups
    if all_summary_rows:
        pd.DataFrame(all_summary_rows).to_csv(os.path.join(args.out, "summary_across_groups.csv"), index=False)

    print(f"\nDone. Outputs written to: {args.out}")
    print("Per group you'll find:")
    print("  group_name/replicates/rep*/timeseries_metrics.csv")
    print("  group_name/aggregate/*_aggregate.csv + plots")
    print("And a global summary: summary_across_groups.csv")


if __name__ == "__main__":
    main()
