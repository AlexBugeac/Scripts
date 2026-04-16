"""
msm_its_manual.py
Compute implied timescales manually without multiprocessing
Loops over lag times sequentially — no hanging
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_COMP = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_comparative'
OUTDIR    = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_validation'

SYSTEMS = {
    '8RJJ-native':    '#E74C3C',
    '8RJJ-SS411-424': '#2ECC71',
    '8RJJ-SS424-429': '#3498DB',
}

LAGS = [1, 2, 5, 10, 20, 50, 100, 200, 500]
NITS = 4  # number of timescales to track

if __name__ == '__main__':
    import pyemma.msm as msm

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (sys_name, color) in zip(axes, SYSTEMS.items()):
        print(f"\n{sys_name}...")
        dtrajs = list(np.load(
            os.path.join(BASE_COMP, sys_name, 'dtrajs.npy'),
            allow_pickle=True))

        timescales = {i: [] for i in range(NITS)}
        valid_lags = []

        for lag in LAGS:
            print(f"  lag={lag}...", end=' ', flush=True)
            try:
                M = msm.estimate_markov_model(dtrajs, lag=lag, reversible=True)
                ts = M.timescales()[:NITS] * 10 / 1000  # frames -> ns
                for i in range(NITS):
                    timescales[i].append(ts[i] if i < len(ts) else np.nan)
                valid_lags.append(lag)
                print(f"ts1={ts[0]:.1f} ns")
            except Exception as e:
                print(f"failed: {e}")
                for i in range(NITS):
                    timescales[i].append(np.nan)

        # Plot
        lag_ps = [l * 10 for l in valid_lags]  # convert to ps
        colors_ts = ['blue', 'red', 'green', 'orange']
        for i in range(NITS):
            ts_arr = np.array(timescales[i])
            valid = ~np.isnan(ts_arr)
            if valid.any():
                ax.semilogy(np.array(lag_ps)[valid], ts_arr[valid],
                           'o-', color=colors_ts[i], lw=1.5, ms=5,
                           label=f'TS {i+1}')

        ax.axvline(500, color='red', ls='--', lw=1, label='lag=500ps')
        ax.set_xlabel('Lag time (ps)')
        ax.set_ylabel('Implied timescale (ns)')
        ax.set_title(sys_name)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle('Implied timescales — manual computation (no multiprocessing)')
    plt.tight_layout()
    out = os.path.join(OUTDIR, 'its_manual.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved: {out}")
