"""
msm_strided_v2.py
MSM on strided DCDs — separate TICA per system
ITS validation + MSM + PCCA
"""

import os
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_OUT = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_strided'
HAIRPIN_CORE = list(range(8, 20))
DT_NS = 0.5  # 500 ps per frame

SYSTEMS = {
    '8RJJ-native': {
        'top':   '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-native/8RJJ-native_05_minimized.pdb',
        'trajs': [f'/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/salieri-data/replica_0{i}_s50_combined.dcd' for i in [1,2,3]],
        'color': '#E74C3C',
    },
    '8RJJ-SS411-424': {
        'top':   '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-m_I411C_S424C_db411-424/8RJJ-m_I411C_S424C_db411-424_05_minimized.pdb',
        'trajs': [f'/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/albicastro-data/8RJJ-m_I411C_S424C_db411-424/replica_0{i}_s50_combined.dcd' for i in [1,2,3]],
        'color': '#2ECC71',
    },
    '8RJJ-SS424-429': {
        'top':   '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-m_I411C_S424C_db424-429/8RJJ-m_I411C_S424C_db424-429_05_minimized.pdb',
        'trajs': [f'/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/beethoven-data/8RJJ-m_I411C_S424C_db424-429/replica_0{i}_s50_combined.dcd' for i in [1,2,3]],
        'color': '#3498DB',
    },
}

def build_feat(top_path):
    import pyemma.coordinates as coor
    feat = coor.featurizer(top_path)
    feat.add_backbone_torsions(selstr='resid 5 to 22', periodic=False)
    top_md = md.load_topology(top_path)
    hairpin_ca = [a.index for a in top_md.atoms
                  if a.name == 'CA' and a.residue.index in HAIRPIN_CORE]
    ca_pairs = [[hairpin_ca[i], hairpin_ca[j]]
                for i in range(len(hairpin_ca))
                for j in range(i+2, len(hairpin_ca))]
    feat.add_distances(ca_pairs)
    key_pairs = [(10,17),(12,15),(11,18),(9,18)]
    key_ca = []
    for r1, r2 in key_pairs:
        a1 = [a.index for a in top_md.atoms if a.name=='CA' and a.residue.index==r1]
        a2 = [a.index for a in top_md.atoms if a.name=='CA' and a.residue.index==r2]
        if a1 and a2:
            key_ca.append([a1[0], a2[0]])
    if key_ca:
        feat.add_distances(key_ca)
    return feat

if __name__ == '__main__':
    import pyemma
    import pyemma.coordinates as coor
    import pyemma.msm as msm

    its_results = {}
    msm_results = {}

    for sys_name, cfg in SYSTEMS.items():
        print(f"\n{'='*60}")
        print(f"{sys_name}")
        print(f"{'='*60}")

        OUTDIR = os.path.join(BASE_OUT, sys_name)
        os.makedirs(OUTDIR, exist_ok=True)

        trajs = [t for t in cfg['trajs'] if os.path.exists(t)]
        feat = build_feat(cfg['top'])
        print(f"  Trajectories: {len(trajs)}, Features: {feat.dimension()}")

        # Load
        data = coor.load(trajs, features=feat, chunksize=5000, n_jobs=1)
        n = sum(len(d) for d in data)
        print(f"  Frames: {n} ({n*DT_NS/1000:.2f} us)")
        del feat

        # TICA
        tica = coor.tica(data, lag=10, dim=5, kinetic_map=True, n_jobs=1)
        tica_output = tica.get_output()
        tica_cat = np.concatenate(tica_output)
        np.save(os.path.join(OUTDIR, 'tica_output.npy'), tica_cat)
        print(f"  TICA cumvar (top 3): {tica.cumvar[:3]}")

        # Landscape
        fig, ax = plt.subplots(figsize=(7, 6))
        pyemma.plots.plot_free_energy(tica_cat[:,0], tica_cat[:,1], ax=ax)
        ax.set_xlabel('TICA IC1')
        ax.set_ylabel('TICA IC2')
        ax.set_title(f'{sys_name} — strided landscape')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'landscape.png'), dpi=150)
        plt.close()

        # Clustering
        cluster = coor.cluster_kmeans(tica_output, k=100, max_iter=100,
                                      stride=1, n_jobs=1)
        dtrajs = cluster.dtrajs
        print(f"  Clusters: {cluster.n_clusters}")

        # ITS
        print("  Computing ITS...")
        LAGS = [1, 2, 5, 10, 20, 50, 100, 200]
        ts_per_lag = []
        valid_lags = []
        for lag in LAGS:
            try:
                M = msm.estimate_markov_model(dtrajs, lag=lag, reversible=True)
                ts = M.timescales()[:4] * DT_NS
                ts_per_lag.append(ts)
                valid_lags.append(lag * DT_NS)
                print(f"    lag={lag*DT_NS:.0f}ns ts1={ts[0]:.1f}ns")
            except Exception as e:
                print(f"    lag={lag} failed: {e}")

        its_results[sys_name] = (valid_lags, ts_per_lag)

        # Final MSM at lag=20 frames (10 ns)
        MSM_LAG = 20
        M = msm.estimate_markov_model(dtrajs, lag=MSM_LAG, reversible=True)
        N_MACRO = 4
        M.pcca(N_MACRO)
        print(f"  MSM lag={MSM_LAG*DT_NS:.0f}ns: {M.nstates} states")
        print(f"  Timescales (ns): {M.timescales()[:4]*DT_NS}")

        pops = [M.pi[M.metastable_sets[i]].sum() for i in range(N_MACRO)]
        print(f"  Populations: {[f'{p:.3f}' for p in pops]}")
        msm_results[sys_name] = {'pops': pops, 'ts': M.timescales()[:4]*DT_NS,
                                  'M': M, 'cluster': cluster, 'tica_cat': tica_cat}

        with open(os.path.join(OUTDIR, 'summary.txt'), 'w') as f:
            f.write(f"=== {sys_name} strided MSM ===\n")
            f.write(f"Frames: {n} ({n*DT_NS/1000:.2f} us)\n")
            f.write(f"MSM lag: {MSM_LAG*DT_NS:.0f} ns\n")
            f.write(f"Active states: {M.nstates}\n")
            f.write(f"Timescales (ns): {M.timescales()[:4]*DT_NS}\n")
            for i in range(N_MACRO):
                f.write(f"Macrostate {i}: {pops[i]:.3f}\n")

        del data, tica, tica_output, cluster, dtrajs
        import gc; gc.collect()

    # ─── ITS comparison plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_ts = ['blue', 'red', 'green', 'orange']
    for ax, (sys_name, cfg) in zip(axes, SYSTEMS.items()):
        valid_lags, ts_per_lag = its_results[sys_name]
        for i in range(4):
            ts_i = [ts[i] if i < len(ts) else np.nan for ts in ts_per_lag]
            ax.semilogy(valid_lags, ts_i, 'o-', color=colors_ts[i],
                       lw=1.5, ms=5, label=f'TS {i+1}')
        ax.axvline(10, color='red', ls='--', lw=1, label='lag=10ns')
        ax.set_xlabel('Lag time (ns)')
        ax.set_ylabel('Implied timescale (ns)')
        ax.set_title(sys_name)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.suptitle('ITS — strided (500 ps/frame), separate TICA per system')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'its_strided_v2.png'), dpi=150)
    plt.close()
    print("\n-> its_strided_v2.png saved")

    # ─── Population comparison ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(4)
    width = 0.25
    for i, (sys_name, cfg) in enumerate(SYSTEMS.items()):
        ax.bar(x + i*width, msm_results[sys_name]['pops'], width,
               label=sys_name, color=cfg['color'], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'State {i}' for i in range(4)])
    ax.set_ylabel('Equilibrium population')
    ax.set_title('Macrostate populations — strided MSM (lag=10 ns)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'populations_strided_v2.png'), dpi=150)
    plt.close()
    print("-> populations_strided_v2.png saved")

    print("\n=== Done ===")
