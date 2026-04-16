"""
msm_from_cluster.py
Run MSM + PCCA from saved dtrajs.npy — skips featurization, TICA, clustering
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUTDIR    = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_output_full'
DTRAJS    = os.path.join(OUTDIR, 'dtrajs.npy')
TICA_OUT  = os.path.join(OUTDIR, 'tica_output.npy')

if __name__ == '__main__':
    import pyemma
    import pyemma.msm as msm
    import pyemma.plots as mplt

    print("Loading saved dtrajs and TICA output...")
    dtrajs   = list(np.load(DTRAJS, allow_pickle=True))
    tica_cat = np.load(TICA_OUT)
    print(f"  dtrajs: {len(dtrajs)} trajectories")
    print(f"  tica:   {tica_cat.shape}")

    # ─── Implied timescales ───────────────────────────────────────────────────
    print("\n=== Implied timescales (n_jobs=4) ===")
    lags = [1, 2, 5, 10, 20, 50, 100]
    its  = msm.its(dtrajs, lags=lags, nits=5, reversible=True, n_jobs=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    mplt.plot_implied_timescales(its, ax=ax, units='frames', dt=1)
    ax.set_xlabel('Lag time (frames, 1 frame = 10 ps)')
    ax.set_ylabel('Implied timescale (frames)')
    ax.set_title('Implied timescales — choose lag where lines plateau')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'implied_timescales.png'), dpi=150)
    plt.close()
    print("  -> implied_timescales.png saved")

    # ─── MSM ──────────────────────────────────────────────────────────────────
    MSM_LAG = 50
    print(f"\n=== MSM (lag={MSM_LAG} frames = {MSM_LAG*10} ps) ===")
    M = msm.estimate_markov_model(dtrajs, lag=MSM_LAG, reversible=True)
    print(f"  Active states: {M.nstates}")
    print(f"  Top 5 timescales (ns): {M.timescales()[:5]*10/1000}")

    # ─── PCCA+ ────────────────────────────────────────────────────────────────
    N_MACRO = 4
    print(f"\n=== PCCA+ ({N_MACRO} macrostates) ===")
    M.pcca(N_MACRO)

    fig, axes = plt.subplots(1, N_MACRO, figsize=(4*N_MACRO, 4))
    for i, ax in enumerate(axes):
        pyemma.plots.plot_free_energy(
            tica_cat[:, 0], tica_cat[:, 1], ax=ax, cbar=False
        )
        macro_states = M.metastable_sets[i]
        pop = M.pi[macro_states].sum()
        ax.set_title(f'Macrostate {i}\npop={pop:.3f}')
        ax.set_xlabel('IC1')
        ax.set_ylabel('IC2' if i == 0 else '')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'macrostates.png'), dpi=150)
    plt.close()
    print("  -> macrostates.png saved")

    # ─── Summary ──────────────────────────────────────────────────────────────
    with open(os.path.join(OUTDIR, 'msm_summary.txt'), 'w') as f:
        f.write("=== Full MSM Summary ===\n")
        f.write(f"MSM lag: {MSM_LAG} frames ({MSM_LAG*10} ps)\n")
        f.write(f"Active states: {M.nstates}\n")
        f.write(f"Top 5 timescales (ns): {M.timescales()[:5]*10/1000}\n")
        f.write(f"Macrostates: {N_MACRO}\n")
        for i in range(N_MACRO):
            pop = M.pi[M.metastable_sets[i]].sum()
            f.write(f"  Macrostate {i}: population {pop:.3f}\n")

    print("\n=== Done ===")
    print(f"Results in {OUTDIR}/")
