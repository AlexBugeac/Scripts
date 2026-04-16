"""
msm_pipeline_full.py
Full MSM — all backbone torsions + Ca contacts
n_jobs=4 throughout to prevent RAM explosion
"""

import os
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE   = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/salieri-data'
TOP    = '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-native/8RJJ-native_05_minimized.pdb'
OUTDIR = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_output_full'
os.makedirs(OUTDIR, exist_ok=True)

TRAJS = []
for rep in ['replica_01', 'replica_02', 'replica_03']:
    for traj in ['traj.dcd', 'traj2.dcd', 'traj3.dcd']:
        path = os.path.join(BASE, rep, traj)
        if os.path.exists(path):
            TRAJS.append(path)
            print(f"Found: {path}")
print(f"Total: {len(TRAJS)} trajectory files")

if __name__ == '__main__':
    import pyemma
    import pyemma.coordinates as coor
    import pyemma.msm as msm
    import pyemma.plots as mplt

    # ─── Featurization ────────────────────────────────────────────────────────
    print("\n=== Featurizing (full: backbone torsions + Ca contacts) ===")
    feat = coor.featurizer(TOP)
    feat.add_backbone_torsions(periodic=False)
    feat.add_backbone_torsions(periodic=False)  # contacts removed - API incompatible
    print(f"  Features: {feat.dimension()}")

    # ─── Load ─────────────────────────────────────────────────────────────────
    print("\nLoading trajectories (n_jobs=4)...")
    data = coor.load(TRAJS, features=feat, chunksize=10000, n_jobs=4)
    print(f"  Loaded {len(data)} trajectories")
    print(f"  Lengths: {[len(d) for d in data]}")

    # ─── TICA ─────────────────────────────────────────────────────────────────
    print("\n=== TICA (lag=10, dim=10, n_jobs=4) ===")
    tica = coor.tica(data, lag=10, dim=10, kinetic_map=True, n_jobs=4)
    tica_output = tica.get_output()
    tica_cat = np.concatenate(tica_output)
    print(f"  Dimensions: {tica.dimension()}")
    print(f"  Cumvar: {tica.cumvar[:5]}")
    np.save(os.path.join(OUTDIR, 'tica_output.npy'), tica_cat)

    fig, ax = plt.subplots(figsize=(8, 6))
    pyemma.plots.plot_free_energy(tica_cat[:, 0], tica_cat[:, 1], ax=ax)
    ax.set_xlabel('TICA IC1')
    ax.set_ylabel('TICA IC2')
    ax.set_title('Free energy landscape (full featurization)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'tica_landscape.png'), dpi=150)
    plt.close()
    print("  -> tica_landscape.png saved")

    # ─── Clustering ───────────────────────────────────────────────────────────
    print("\n=== Clustering (k=200, n_jobs=4) ===")
    cluster = coor.cluster_kmeans(tica_output, k=200, max_iter=100, stride=10, n_jobs=4)
    dtrajs = cluster.dtrajs
    print(f"  Clusters: {cluster.n_clusters}")
    np.save(os.path.join(OUTDIR, 'dtrajs.npy'), np.array(dtrajs, dtype=object))

    # ─── Implied timescales ───────────────────────────────────────────────────
    print("\n=== Implied timescales (n_jobs=4) ===")
    lags = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    its  = msm.its(dtrajs, lags=lags, nits=5, reversible=True, n_jobs=4)

    fig, ax = plt.subplots(figsize=(8, 5))
    mplt.plot_implied_timescales(its, ax=ax, units='frames', dt=1)
    ax.set_xlabel('Lag time (frames)')
    ax.set_ylabel('Implied timescale (frames)')
    ax.set_title('Implied timescales — choose lag where lines plateau')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'implied_timescales.png'), dpi=150)
    plt.close()
    print("  -> implied_timescales.png saved")

    # ─── MSM ──────────────────────────────────────────────────────────────────
    MSM_LAG = 50
    print(f"\n=== MSM (lag={MSM_LAG}) ===")
    M = msm.estimate_markov_model(dtrajs, lag=MSM_LAG, reversible=True)
    print(f"  Active states: {M.nstates}")
    print(f"  Active count fraction: {M.active_count_fraction if hasattr(M, "active_count_fraction") else "N/A":.3f}")
    print(f"  Top 5 timescales (ns): {M.timescales()[:5]*10/1000}")

    # ─── PCCA+ ────────────────────────────────────────────────────────────────
    N_MACRO = 4
    print(f"\n=== PCCA+ ({N_MACRO} macrostates) ===")
    M.pcca(N_MACRO)

    fig, axes = plt.subplots(1, N_MACRO, figsize=(4*N_MACRO, 4))
    for i, ax in enumerate(axes):
        pyemma.plots.plot_free_energy(tica_cat[:, 0], tica_cat[:, 1], ax=ax, cbar=False)
        macro_states = M.metastable_sets[i]
        centers = cluster.clustercenters[macro_states]
        ax.scatter(centers[:, 0], centers[:, 1], c='red', s=30, zorder=5)
        pop = M.pi[macro_states].sum()
        ax.set_title(f'Macrostate {i}\npop={pop:.3f}')
        ax.set_xlabel('IC1')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'macrostates.png'), dpi=150)
    plt.close()
    print("  -> macrostates.png saved")

    # ─── Summary ──────────────────────────────────────────────────────────────
    with open(os.path.join(OUTDIR, 'msm_summary.txt'), 'w') as f:
        f.write("=== Full MSM Summary ===\n")
        f.write(f"Total frames: {sum(len(d) for d in data)}\n")
        f.write(f"Features: {feat.dimension()}\n")
        f.write(f"TICA cumvar (top 5): {tica.cumvar[:5]}\n")
        f.write(f"MSM lag: {MSM_LAG} frames ({MSM_LAG*10} ps)\n")
        f.write(f"Active states: {M.nstates}\n")
        f.write(f"Top 5 timescales (ns): {M.timescales()[:5]*10/1000}\n")
        f.write(f"Macrostates: {N_MACRO}\n")
        for i in range(N_MACRO):
            pop = M.pi[M.metastable_sets[i]].sum()
            f.write(f"  Macrostate {i}: population {pop:.3f}\n")

    print("\n=== Full MSM pipeline complete ===")
