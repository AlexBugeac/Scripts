"""
msm_validate_extract.py
1. ITS validation for lag time selection
2. Test N=5,6 macrostates
3. Extract representative structures from each macrostate
"""

import os
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_JOINT = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_joint'
BASE_COMP  = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_comparative'
OUTDIR     = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_validation'
os.makedirs(OUTDIR, exist_ok=True)

SYSTEMS = {
    '8RJJ-native': {
        'top':  '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-native/8RJJ-native_05_minimized.pdb',
        'base': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/salieri-data',
        'color': '#E74C3C',
    },
    '8RJJ-SS411-424': {
        'top':  '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-m_I411C_S424C_db411-424/8RJJ-m_I411C_S424C_db411-424_05_minimized.pdb',
        'base': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/albicastro-data/8RJJ-m_I411C_S424C_db411-424',
        'color': '#2ECC71',
    },
    '8RJJ-SS424-429': {
        'top':  '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-m_I411C_S424C_db424-429/8RJJ-m_I411C_S424C_db424-429_05_minimized.pdb',
        'base': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/beethoven-data/8RJJ-m_I411C_S424C_db424-429',
        'color': '#3498DB',
    },
}

if __name__ == '__main__':
    import pyemma
    import pyemma.coordinates as coor
    import pyemma.msm as msm

    # ─── 1. ITS validation ────────────────────────────────────────────────────
    print("=== 1. Implied timescales validation ===")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (sys_name, cfg) in zip(axes, SYSTEMS.items()):
        print(f"\n  {sys_name}...")
        dtrajs_path = os.path.join(BASE_COMP, sys_name, 'dtrajs.npy')
        dtrajs = list(np.load(dtrajs_path, allow_pickle=True))

        lags = [1, 2, 5, 10, 20, 50]
        its = msm.its(dtrajs, lags=lags, nits=3, reversible=True, n_jobs=1)

        pyemma.plots.plot_implied_timescales(its, ax=ax, units='frames', dt=1)
        ax.set_title(f'{sys_name}')
        ax.set_xlabel('Lag time (frames, 1=10ps)')
        ax.set_ylabel('Implied timescale (frames)')
        ax.axvline(50, color='red', ls='--', lw=1, label='lag=50')
        ax.legend(fontsize=8)
        print(f"  Done")

    plt.suptitle('Implied timescales — lag time validation', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'implied_timescales_validation.png'), dpi=150)
    plt.close()
    print("  -> implied_timescales_validation.png saved")

    # ─── 2. Test N macrostates ────────────────────────────────────────────────
    print("\n=== 2. Testing N macrostates for 8RJJ-native ===")
    dtrajs_native = list(np.load(
        os.path.join(BASE_COMP, '8RJJ-native', 'dtrajs.npy'), allow_pickle=True))
    tica_native = np.load(os.path.join(BASE_JOINT, 'tica_8RJJ-native.npy'))

    M_native = msm.estimate_markov_model(dtrajs_native, lag=50, reversible=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for n_macro, ax in zip([3, 4, 5, 6], axes):
        M_native.pcca(n_macro)
        pyemma.plots.plot_free_energy(tica_native[:,0], tica_native[:,1],
                                       ax=ax, cbar=False)
        colors = ['red','lime','blue','orange','cyan','magenta']
        for i in range(n_macro):
            macro_idx = M_native.metastable_sets[i]
            pop = M_native.pi[macro_idx].sum()
            ax.set_title(f'N={n_macro}')
            ax.set_xlabel('IC1')
        ax.set_ylabel('IC2' if ax == axes[0] else '')
    plt.suptitle('8RJJ-native: effect of number of macrostates')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'macrostate_number_test.png'), dpi=150)
    plt.close()
    print("  -> macrostate_number_test.png saved")

    # ─── 3. Extract representative structures ─────────────────────────────────
    print("\n=== 3. Extracting representative structures ===")
    struct_dir = os.path.join(OUTDIR, 'macrostate_structures')
    os.makedirs(struct_dir, exist_ok=True)

    # Use 8RJJ-native with N=4
    M_native.pcca(4)
    tica_native = np.load(os.path.join(BASE_JOINT, 'tica_8RJJ-native.npy'))

    cfg = SYSTEMS['8RJJ-native']
    TRAJS = []
    for rep in ['replica_01', 'replica_02', 'replica_03']:
        for traj in ['traj.dcd', 'traj2.dcd', 'traj3.dcd']:
            path = os.path.join(cfg['base'], rep, traj)
            if os.path.exists(path):
                TRAJS.append(path)

    # Get macrostate assignment per frame
    dtraj_concat = np.concatenate(dtrajs_native)
    macro_assign = np.zeros(len(dtraj_concat), dtype=int) - 1
    for i in range(4):
        for microstate in M_native.metastable_sets[i]:
            macro_assign[dtraj_concat == microstate] = i

    print(f"  Total frames: {len(macro_assign)}")
    for i in range(4):
        n = (macro_assign == i).sum()
        pop = M_native.pi[M_native.metastable_sets[i]].sum()
        print(f"  Macrostate {i}: {n} frames ({pop:.3f} population)")

    # Find frame closest to centroid of each macrostate in TICA space
    for macro_i in range(4):
        mask = macro_assign == macro_i
        if mask.sum() == 0:
            continue
        tica_macro = tica_native[mask]
        centroid = tica_macro.mean(axis=0)
        dist = np.linalg.norm(tica_native - centroid, axis=1)
        dist[~mask] = np.inf
        closest_frame = np.argmin(dist)

        # Find which trajectory and local frame
        cumulative = 0
        for traj_path in TRAJS:
            t = md.load(traj_path, top=cfg['top'])
            n = t.n_frames
            if closest_frame < cumulative + n:
                local_frame = closest_frame - cumulative
                frame = t[local_frame]
                out_pdb = os.path.join(struct_dir, f'macrostate_{macro_i}_representative.pdb')
                frame.save_pdb(out_pdb)
                pop = M_native.pi[M_native.metastable_sets[macro_i]].sum()
                print(f"  Macrostate {macro_i} (pop={pop:.3f}): saved {out_pdb}")
                break
            cumulative += n

    print(f"\n=== Validation complete. Results in {OUTDIR}/ ===")
