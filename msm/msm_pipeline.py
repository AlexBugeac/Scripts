#!/usr/bin/env python3
"""
msm_pipeline.py — Unified MSM pipeline for implicit-solvent MD trajectories.

Replaces: msm_pipeline_full.py, msm_pipeline_focused.py, msm_joint_msm.py,
          msm_joint_normalized.py, msm_joint_tica.py, msm_all_systems.py,
          msm_full_all.py, msm_strided.py, msm_strided_v2.py,
          msm_final.py, plot_comparative_rmsf.py, project_crystals_joint.py

──────────────────────────────────────────────────────────────────────────────
SINGLE SYSTEM (one topology, one set of replicas):

  python msm_pipeline.py \\
    --topology  8RJJ-native_05_minimized.pdb \\
    --trajs     replica_01/traj*.dcd replica_02/traj*.dcd replica_03/traj*.dcd \\
    --outdir    results/msm_native \\
    --feat      focused \\
    --lag       50 \\
    --n-macro   4

JOINT ANALYSIS (multiple systems on shared TICA space):

  python msm_pipeline.py \\
    --systems   systems.yaml \\
    --outdir    results/msm_joint \\
    --feat      focused \\
    --normalize \\           # z-score per system before joint TICA
    --lag       20 \\
    --n-macro   4

──────────────────────────────────────────────────────────────────────────────
systems.yaml format:
  - name:   "8RJJ-native"
    top:    "/path/to/8RJJ-native_05_minimized.pdb"
    trajs:  ["/path/rep01/traj_s50.dcd", "/path/rep02/traj_s50.dcd"]
    color:  "#E74C3C"        # optional; for plots

  - name:   "8RJJ-db411-424"
    top:    "/path/to/8RJJ-db411-424_05_minimized.pdb"
    trajs:  ["/path/rep01/traj_s50.dcd", ...]
    color:  "#2ECC71"

──────────────────────────────────────────────────────────────────────────────
Featurization modes  (--feat):
  full      backbone torsions (all residues) + Cα contacts
  focused   backbone torsions + Cα distances for a residue window
            (set --feat-window, e.g. "5:22")
            + optional key contact pairs (--key-pairs)
  custom    backbone torsions only (safe fallback if contacts cause issues)

Output (per run):
  tica_output.npy          raw TICA projections
  tica_landscape.png       2D free energy surface
  implied_timescales.png   ITS vs lag — review before trusting MSM
  macrostates.png          PCCA+ macrostate overlay on FES
  msm_summary.txt          populations, timescales, metadata

Joint-only additional output:
  joint_landscapes.png     per-system FES on shared TICA axes
  populations_bar.png      macrostate population comparison
  tica_<name>.npy          per-system TICA projections
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mdtraj as md

warnings.filterwarnings("ignore")


# ── Featurization ─────────────────────────────────────────────────────────────

def build_featurizer(
    top_path: str,
    mode:     str,
    window:   tuple[int, int] | None = None,
    key_pairs: list[tuple[int, int]] | None = None,
):
    """Return a PyEMMA featurizer for the given mode."""
    import pyemma.coordinates as coor

    feat = coor.featurizer(top_path)

    if mode == "full":
        feat.add_backbone_torsions(periodic=False)
        # Cα contacts (all pairs, sep ≥ 3)
        top_md = md.load_topology(top_path)
        ca = [a.index for a in top_md.atoms if a.name == "CA"]
        pairs = [[ca[i], ca[j]] for i in range(len(ca)) for j in range(i + 3, len(ca))]
        if pairs:
            feat.add_distances(pairs)

    elif mode == "focused":
        lo, hi = window if window else (5, 22)
        feat.add_backbone_torsions(selstr=f"resid {lo} to {hi}", periodic=False)

        top_md  = md.load_topology(top_path)
        win_res = list(range(lo, hi + 1))
        ca_all  = {a.residue.index: a.index
                   for a in top_md.atoms if a.name == "CA"}
        ca_win  = [ca_all[r] for r in win_res if r in ca_all]
        pairs   = [[ca_win[i], ca_win[j]]
                   for i in range(len(ca_win))
                   for j in range(i + 2, len(ca_win))]
        if pairs:
            feat.add_distances(pairs)

        if key_pairs:
            kp = []
            for r1, r2 in key_pairs:
                if r1 in ca_all and r2 in ca_all:
                    kp.append([ca_all[r1], ca_all[r2]])
            if kp:
                feat.add_distances(kp)

    elif mode == "custom":
        feat.add_backbone_torsions(periodic=False)

    else:
        sys.exit(f"ERROR: unknown featurization mode '{mode}'. "
                 "Choose from: full | focused | custom")

    return feat


# ── Data loading + optional normalization ─────────────────────────────────────

def load_and_normalize(
    trajs:     list[str],
    top_path:  str,
    feat_mode: str,
    window:    tuple[int, int] | None,
    key_pairs: list[tuple[int, int]] | None,
    normalize: bool,
    n_jobs:    int,
    chunksize: int,
) -> tuple[list[np.ndarray], int]:
    """Load trajectories, optionally z-score normalize. Return (data, n_features)."""
    import pyemma.coordinates as coor

    trajs_exist = [t for t in trajs if Path(t).exists()]
    if not trajs_exist:
        sys.exit(f"ERROR: none of the trajectory files exist.\n  {trajs}")
    if len(trajs_exist) < len(trajs):
        missing = set(trajs) - set(trajs_exist)
        print(f"  WARNING: {len(missing)} trajectory file(s) not found — skipping")

    feat = build_featurizer(top_path, feat_mode, window, key_pairs)
    print(f"  Features: {feat.dimension()}")

    data = coor.load(trajs_exist, features=feat, chunksize=chunksize, n_jobs=n_jobs)

    if normalize:
        all_frames = np.concatenate(data)
        mean = all_frames.mean(axis=0)
        std  = all_frames.std(axis=0)
        std[std < 1e-8] = 1.0
        data = [(d - mean) / std for d in data]
        print(f"  Normalized (z-score). Mean range: [{mean.min():.2f}, {mean.max():.2f}]")

    return data, feat.dimension()


# ── TICA + clustering ─────────────────────────────────────────────────────────

def run_tica(data, lag: int, dim: int, n_jobs: int):
    import pyemma.coordinates as coor
    print(f"  TICA: lag={lag}, dim={dim}")
    tica = coor.tica(data, lag=lag, dim=dim, kinetic_map=True, n_jobs=n_jobs)
    print(f"  Cumvar (top 3): {tica.cumvar[:3]}")
    return tica


def run_clustering(tica_output, k: int, n_jobs: int):
    import pyemma.coordinates as coor
    print(f"  K-means: k={k}")
    cluster = coor.cluster_kmeans(
        tica_output, k=k, max_iter=100, stride=10, n_jobs=n_jobs
    )
    return cluster


def run_its(dtrajs, lags, n_jobs: int):
    import pyemma.msm as msm
    its = msm.its(dtrajs, lags=lags, nits=5, reversible=True, n_jobs=n_jobs)
    return its


def run_msm(dtrajs, lag: int, n_macro: int):
    import pyemma.msm as msm
    M = msm.estimate_markov_model(dtrajs, lag=lag, reversible=True)
    M.pcca(n_macro)
    return M


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_landscape(tica_cat, outpath: Path, title: str = "") -> None:
    import pyemma
    fig, ax = plt.subplots(figsize=(7, 5))
    pyemma.plots.plot_free_energy(tica_cat[:, 0], tica_cat[:, 1], ax=ax)
    ax.set_xlabel("TICA IC1")
    ax.set_ylabel("TICA IC2")
    ax.set_title(title or "Free energy landscape")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  -> {outpath.name}")


def plot_its(its, dt_ps: float, outpath: Path) -> None:
    import pyemma.plots as mplt
    fig, ax = plt.subplots(figsize=(7, 4))
    mplt.plot_implied_timescales(its, ax=ax, units="frames", dt=1)
    ax.set_xlabel("Lag time (frames)")
    ax.set_ylabel(f"Implied timescale (frames × {dt_ps:.0f} ps)")
    ax.set_title("Implied timescales — plateau = good Markovian lag")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  -> {outpath.name}")


def plot_macrostates(tica_cat, M, cluster, n_macro: int, outpath: Path, title: str = "") -> None:
    import pyemma
    colors = ["red", "lime", "blue", "orange", "purple", "cyan"]
    fig, axes = plt.subplots(1, n_macro, figsize=(4 * n_macro, 4), sharey=True)
    if n_macro == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        pyemma.plots.plot_free_energy(
            tica_cat[:, 0], tica_cat[:, 1], ax=ax, cbar=False
        )
        mc  = cluster.clustercenters[M.metastable_sets[i]]
        pop = M.pi[M.metastable_sets[i]].sum()
        ax.scatter(mc[:, 0], mc[:, 1], c=colors[i % len(colors)], s=30, zorder=5)
        ax.set_title(f"State {i}\npop={pop:.3f}")
        ax.set_xlabel("IC1")
        if i == 0:
            ax.set_ylabel("IC2")
    if title:
        plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  -> {outpath.name}")


# ── Single-system pipeline ────────────────────────────────────────────────────

def run_single(args, outdir: Path) -> None:
    import pyemma
    outdir.mkdir(parents=True, exist_ok=True)

    trajs = []
    for pattern in args.trajs:
        matched = sorted(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]
        trajs.extend([str(p) for p in matched])
    if not trajs:
        sys.exit("ERROR: no trajectory files matched --trajs patterns")

    print(f"\n{'='*60}")
    print(f"Single-system MSM: {len(trajs)} trajectory file(s)")
    print(f"Topology : {args.topology}")
    print(f"Outdir   : {outdir}")
    print(f"{'='*60}")

    # Parse window
    window = None
    if args.feat_window:
        lo, hi = map(int, args.feat_window.split(":"))
        window = (lo, hi)

    # Parse key pairs
    key_pairs = None
    if args.key_pairs:
        key_pairs = [tuple(map(int, p.split(","))) for p in args.key_pairs]

    print("\n=== Featurize + Load ===")
    data, n_feat = load_and_normalize(
        trajs, args.topology, args.feat, window, key_pairs,
        normalize=False, n_jobs=args.n_jobs, chunksize=args.chunksize,
    )

    print("\n=== TICA ===")
    tica = run_tica(data, lag=args.tica_lag, dim=args.tica_dim, n_jobs=args.n_jobs)
    tica_out = tica.get_output()
    tica_cat = np.concatenate(tica_out)
    np.save(outdir / "tica_output.npy", tica_cat)
    tica.save(str(outdir / "tica_model.pyemma"), overwrite=True)
    plot_landscape(tica_cat, outdir / "tica_landscape.png")

    print("\n=== Clustering ===")
    cluster = run_clustering(tica_out, k=args.n_clusters, n_jobs=args.n_jobs)
    dtrajs  = cluster.dtrajs
    np.save(outdir / "dtrajs.npy", np.array(dtrajs, dtype=object))

    print("\n=== Implied timescales ===")
    its = run_its(dtrajs, lags=args.its_lags, n_jobs=args.n_jobs)
    plot_its(its, dt_ps=args.dt_ps, outpath=outdir / "implied_timescales.png")

    print(f"\n=== MSM (lag={args.lag}) ===")
    M = run_msm(dtrajs, lag=args.lag, n_macro=args.n_macro)
    print(f"  Active states: {M.nstates}")
    print(f"  Top timescales (ns): {M.timescales()[:5] * args.dt_ps / 1e6}")
    plot_macrostates(tica_cat, M, cluster, args.n_macro, outdir / "macrostates.png")

    # Summary
    with (outdir / "msm_summary.txt").open("w") as f:
        f.write(f"=== MSM Summary ===\n")
        f.write(f"Topology: {args.topology}\n")
        f.write(f"Trajectories: {len(trajs)}\n")
        f.write(f"Total frames: {sum(len(d) for d in data)}\n")
        f.write(f"Features: {n_feat}\n")
        f.write(f"Featurization: {args.feat}\n")
        f.write(f"TICA lag: {args.tica_lag} | dim: {args.tica_dim}\n")
        f.write(f"Clusters: {args.n_clusters} | MSM lag: {args.lag}\n")
        f.write(f"Macrostates: {args.n_macro}\n\n")
        f.write(f"TICA cumvar (top 5): {tica.cumvar[:5]}\n")
        f.write(f"Active states: {M.nstates}\n")
        f.write(f"Top 5 timescales (ns): {M.timescales()[:5] * args.dt_ps / 1e6}\n\n")
        for i in range(args.n_macro):
            pop = M.pi[M.metastable_sets[i]].sum()
            f.write(f"Macrostate {i}: population {pop:.4f}\n")

    print(f"\n=== Done → {outdir}/ ===")


# ── Joint pipeline ────────────────────────────────────────────────────────────

def load_systems_yaml(path: str) -> list[dict]:
    try:
        import yaml
    except ImportError:
        sys.exit("ERROR: pyyaml required for --systems. Install: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def run_joint(args, outdir: Path) -> None:
    import pyemma
    import pyemma.coordinates as coor
    import pyemma.msm as msm
    outdir.mkdir(parents=True, exist_ok=True)

    systems = load_systems_yaml(args.systems)
    print(f"\n{'='*60}")
    print(f"Joint MSM: {len(systems)} system(s)")
    print(f"Normalize: {args.normalize}")
    print(f"Outdir   : {outdir}")
    print(f"{'='*60}")

    window = None
    if args.feat_window:
        lo, hi = map(int, args.feat_window.split(":"))
        window = (lo, hi)
    key_pairs = [tuple(map(int, p.split(","))) for p in args.key_pairs] \
        if args.key_pairs else None

    # ── Step 1: Featurize each system ─────────────────────────────────────────
    print("\n=== Step 1: Featurize + load each system ===")
    system_data: dict[str, list[np.ndarray]] = {}
    system_lengths: dict[str, list[int]] = {}

    for cfg in systems:
        name = cfg["name"]
        top  = cfg["top"]
        trajs = [t for t in cfg["trajs"] if Path(t).exists()]
        print(f"\n  {name}  ({len(trajs)} trajectories)")
        data, _ = load_and_normalize(
            trajs, top, args.feat, window, key_pairs,
            normalize=args.normalize, n_jobs=args.n_jobs, chunksize=args.chunksize,
        )
        system_data[name]    = data
        system_lengths[name] = [len(d) for d in data]

    # ── Step 2: Joint TICA ────────────────────────────────────────────────────
    print("\n=== Step 2: Joint TICA ===")
    all_data: list[np.ndarray] = []
    for name in [s["name"] for s in systems]:
        all_data.extend(system_data[name])

    print(f"  Total datasets: {len(all_data)} | "
          f"frames: {sum(len(d) for d in all_data)}")

    tica = coor.tica(all_data, lag=args.tica_lag, dim=args.tica_dim,
                     kinetic_map=True, n_jobs=args.n_jobs)
    print(f"  Cumvar (top 3): {tica.cumvar[:3]}")
    tica.save(str(outdir / "joint_tica.pyemma"), overwrite=True)

    # Project each system
    system_tica: dict[str, np.ndarray] = {}
    idx = 0
    for cfg in systems:
        name    = cfg["name"]
        n_traj  = len(system_lengths[name])
        tc      = np.concatenate(tica.transform(all_data[idx:idx + n_traj]))
        system_tica[name] = tc
        np.save(outdir / f"tica_{name}.npy", tc)
        print(f"  {name}: {tc.shape[0]} frames, "
              f"IC1=[{tc[:,0].min():.2f}, {tc[:,0].max():.2f}], "
              f"IC2=[{tc[:,1].min():.2f}, {tc[:,1].max():.2f}]")
        idx += n_traj

    # ── Step 3: Per-system landscapes ─────────────────────────────────────────
    print("\n=== Step 3: Free energy landscapes ===")
    default_colors = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6"]
    n_sys = len(systems)
    fig, axes = plt.subplots(1, n_sys, figsize=(6 * n_sys, 5),
                             sharex=True, sharey=True)
    if n_sys == 1:
        axes = [axes]

    for ax, cfg in zip(axes, systems):
        name  = cfg["name"]
        color = cfg.get("color", default_colors[systems.index(cfg) % len(default_colors)])
        tc    = system_tica[name]
        pyemma.plots.plot_free_energy(tc[:, 0], tc[:, 1], ax=ax)
        dt_us = sum(system_lengths[name]) * args.dt_ps / 1e9
        ax.set_title(f"{name}\n({dt_us:.1f} µs)")
        ax.set_xlabel("Joint TICA IC1")
        if ax is axes[0]:
            ax.set_ylabel("Joint TICA IC2")

    norm_label = " (normalized)" if args.normalize else ""
    plt.suptitle(f"Free energy landscapes — joint TICA{norm_label}", fontsize=12)
    plt.tight_layout()
    plt.savefig(outdir / "joint_landscapes.png", dpi=150)
    plt.close()
    print("  -> joint_landscapes.png")

    # ── Step 4: Joint clustering ───────────────────────────────────────────────
    print("\n=== Step 4: Joint clustering ===")
    from sklearn.cluster import MiniBatchKMeans
    all_tica = np.concatenate(list(system_tica.values()))
    km = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    km.fit(all_tica[:, :2])
    centers = km.cluster_centers_

    system_dtrajs: dict[str, list[np.ndarray]] = {}
    for cfg in systems:
        name    = cfg["name"]
        tc      = system_tica[name]
        labels  = km.predict(tc[:, :2])
        lengths = system_lengths[name]
        dtrajs  = []
        start   = 0
        for l in lengths:
            dtrajs.append(labels[start:start + l])
            start += l
        system_dtrajs[name] = dtrajs
        print(f"  {name}: {len(np.unique(labels))} unique clusters")

    # ── Step 5: Per-system MSM + PCCA ─────────────────────────────────────────
    print(f"\n=== Step 5: MSM + PCCA (lag={args.lag}) ===")
    results: dict[str, dict] = {}
    macro_colors = ["red", "lime", "blue", "orange", "purple", "cyan"]

    fig2, axes2 = plt.subplots(1, n_sys, figsize=(6 * n_sys, 5),
                               sharex=True, sharey=True)
    if n_sys == 1:
        axes2 = [axes2]

    for ax, cfg in zip(axes2, systems):
        name = cfg["name"]
        print(f"\n  {name}:")
        M = run_msm(system_dtrajs[name], lag=args.lag, n_macro=args.n_macro)
        print(f"    Active states: {M.nstates}")
        ts_ns = M.timescales()[:4] * args.dt_ps / 1e6
        print(f"    Timescales (ns): {ts_ns}")

        pops = []
        for i in range(args.n_macro):
            pop = M.pi[M.metastable_sets[i]].sum()
            pops.append(pop)
            print(f"    Macrostate {i}: {pop:.3f}")

        results[name] = {"pops": pops, "ts_ns": ts_ns}

        tc = system_tica[name]
        pyemma.plots.plot_free_energy(tc[:, 0], tc[:, 1], ax=ax, cbar=False)
        for i in range(args.n_macro):
            mc  = centers[M.metastable_sets[i]]
            pop = pops[i]
            ax.scatter(mc[:, 0], mc[:, 1],
                       c=macro_colors[i % len(macro_colors)], s=30, zorder=5, alpha=0.7)
            ax.annotate(
                f"S{i}\n{pop:.2f}",
                (mc[:, 0].mean(), mc[:, 1].mean()),
                fontsize=8, ha="center", color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor=macro_colors[i % len(macro_colors)], alpha=0.7),
            )
        ax.set_title(name)
        ax.set_xlabel("Joint TICA IC1")
        if ax is axes2[0]:
            ax.set_ylabel("Joint TICA IC2")

    plt.suptitle(
        f"Macrostates — joint TICA{norm_label} (lag={args.lag} frames)", fontsize=11
    )
    plt.tight_layout()
    plt.savefig(outdir / "joint_macrostates.png", dpi=150)
    plt.close()
    print("\n  -> joint_macrostates.png")

    # ── Step 6: Population bar chart ──────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(3 * n_sys + 3, 5))
    x     = np.arange(args.n_macro)
    width = 0.8 / n_sys
    for i, cfg in enumerate(systems):
        name  = cfg["name"]
        color = cfg.get("color", default_colors[i % len(default_colors)])
        ax3.bar(x + i * width, results[name]["pops"], width,
                label=name, color=color, alpha=0.85)
    ax3.set_xticks(x + width * (n_sys - 1) / 2)
    ax3.set_xticklabels([f"State {i}" for i in range(args.n_macro)])
    ax3.set_ylabel("Equilibrium population")
    ax3.set_title(f"Macrostate populations — joint TICA{norm_label}")
    ax3.legend()
    ax3.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(outdir / "populations_bar.png", dpi=150)
    plt.close()
    print("  -> populations_bar.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    with (outdir / "msm_summary.txt").open("w") as f:
        f.write(f"=== Joint MSM Summary ===\n")
        f.write(f"Systems: {[s['name'] for s in systems]}\n")
        f.write(f"Featurization: {args.feat}\n")
        f.write(f"Normalize: {args.normalize}\n")
        f.write(f"TICA lag: {args.tica_lag} | dim: {args.tica_dim}\n")
        f.write(f"Clusters: {args.n_clusters} | MSM lag: {args.lag}\n")
        f.write(f"Macrostates: {args.n_macro}\n")
        f.write(f"dt: {args.dt_ps} ps/frame\n\n")
        f.write(f"TICA cumvar (top 3): {tica.cumvar[:3]}\n\n")
        for name, res in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Timescales (ns): {res['ts_ns']}\n")
            for i, pop in enumerate(res["pops"]):
                f.write(f"  Macrostate {i}: {pop:.4f}\n")
            f.write("\n")

    print(f"\n=== Done → {outdir}/ ===")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified MSM pipeline — single system or joint multi-system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode_grp = ap.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument("--topology", metavar="PDB",
                          help="Single-system topology PDB")
    mode_grp.add_argument("--systems", metavar="YAML",
                          help="YAML file defining multiple systems (joint mode)")

    ap.add_argument("--trajs", nargs="+", metavar="TRAJ",
                    help="Trajectory files / glob patterns (single-system mode)")
    ap.add_argument("--outdir", required=True, metavar="DIR",
                    help="Output directory")

    # Featurization
    ap.add_argument("--feat", default="focused",
                    choices=["full", "focused", "custom"],
                    help="Featurization mode (default: focused)")
    ap.add_argument("--feat-window", default="5:22", metavar="LO:HI",
                    help="Residue window for focused featurization (default: 5:22)")
    ap.add_argument("--key-pairs", nargs="*", metavar="R1,R2",
                    help="Additional key Cα contact pairs for focused mode, e.g. 10,17 12,15")
    ap.add_argument("--normalize", action="store_true",
                    help="Z-score normalize features per system before joint TICA")

    # TICA
    ap.add_argument("--tica-lag", type=int, default=10, metavar="FRAMES",
                    help="TICA lag time in frames (default: 10)")
    ap.add_argument("--tica-dim", type=int, default=5,
                    help="TICA output dimensions (default: 5)")

    # Clustering + MSM
    ap.add_argument("--n-clusters", type=int, default=100, metavar="K",
                    help="k-means cluster centers (default: 100)")
    ap.add_argument("--lag", type=int, default=20, metavar="FRAMES",
                    help="MSM lag time in frames (default: 20)")
    ap.add_argument("--n-macro", type=int, default=4,
                    help="Number of PCCA+ macrostates (default: 4)")
    ap.add_argument("--its-lags", nargs="+", type=int,
                    default=[1, 2, 5, 10, 20, 50, 100, 200, 500],
                    help="Lag values for ITS plot")

    # Performance + timing
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel jobs for PyEMMA (default: 1)")
    ap.add_argument("--chunksize", type=int, default=5000,
                    help="Trajectory loading chunk size (default: 5000)")
    ap.add_argument("--dt-ps", type=float, default=10.0, metavar="PS",
                    help="Frame time step in picoseconds (default: 10 ps = 5000 steps × 2 fs)")

    args = ap.parse_args()

    if args.topology and not args.trajs:
        ap.error("--topology requires --trajs")

    outdir = Path(args.outdir)

    if args.topology:
        run_single(args, outdir)
    else:
        run_joint(args, outdir)


if __name__ == "__main__":
    main()
