"""
msm_strided.py
MSM pipeline using stride-50 combined DCDs (500 ps/frame)
Joint TICA + per-system MSM + ITS validation
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
os.makedirs(BASE_OUT, exist_ok=True)

HAIRPIN_CORE = list(range(8, 20))

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

DT_NS = 0.5  # 500 ps per frame

def build_feat(top_path):
    import pyemma.coordinates as coor
    feat = coor.featurizer(top_path)
    feat.add_backbone_torsions(selstr='resid 5 to 22', periodic=False)
    top_md = md.load_topology(top_path)
    hairpin_ca = [a.index for a in top_md.atoms
                  if a.name == 'CA' and a.residue.index in HAIRPIN_CORE]
    ca_pairs = []
    for i in range(len(hairpin_ca)):
        for j in range(i+2, len(hairpin_ca)):
            ca_pairs.append([hairpin_ca[i], hairpin_ca[j]])
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

    # ─── Step 1: Featurize all systems ────────────────────────────────────────
    print("=== Step 1: Featurizing ===")
    all_data = []
    system_lengths = {}

    for sys_name, cfg in SYSTEMS.items():
        print(f"\n  {sys_name}...")
        feat = build_feat(cfg['top'])
        trajs = [t for t in cfg['trajs'] if os.path.exists(t)]
        print(f"    Trajectories: {len(trajs)}, Features: {feat.dimension()}")
        data = coor.load(trajs, features=feat, chunksize=5000, n_jobs=1)
        n = sum(len(d) for d in data)
        system_lengths[sys_name] = [len(d) for d in data]
        all_data.extend(data)
        print(f"    Frames: {n} ({n*DT_NS/1000:.2f} us)")
        del feat

    # ─── Step 2: Joint TICA ───────────────────────────────────────────────────
    print("\n=== Step 2: Joint TICA ===")
    tica = coor.tica(all_data, lag=10, dim=5, kinetic_map=True, n_jobs=1)
    print(f"  Cumvar (top 3): {tica.cumvar[:3]}")
    tica.save(os.path.join(BASE_OUT, 'tica_model.pyemma'), overwrite=True)

    system_tica = {}
    idx = 0
    for sys_name, cfg in SYSTEMS.items():
        n_trajs = len(system_lengths[sys_name])
        tica_cat = np.concatenate(tica.transform(all_data[idx:idx+n_trajs]))
        system_tica[sys_name] = tica_cat
        np.save(os.path.join(BASE_OUT, f'tica_{sys_name}.npy'), tica_cat)
        idx += n_trajs
        print(f"  {sys_name}: {tica_cat.shape[0]} frames projected")

    # ─── Step 3: Joint clustering ─────────────────────────────────────────────
    print("\n=== Step 3: Joint clustering (k=100) ===")
    from sklearn.cluster import MiniBatchKMeans
    all_tica = np.concatenate(list(system_tica.values()))
    kmeans = MiniBatchKMeans(n_clusters=100, random_state=42, n_init=10)
    kmeans.fit(all_tica[:, :2])

    system_dtrajs = {}
    for sys_name, tica_cat in system_tica.items():
        labels = kmeans.predict(tica_cat[:, :2])
        # Split back into per-trajectory dtrajs
        lengths = system_lengths[sys_name]
        dtrajs = []
        start = 0
        for l in lengths:
            dtrajs.append(labels[start:start+l])
            start += l
        system_dtrajs[sys_name] = dtrajs
        print(f"  {sys_name}: {len(np.unique(labels))} unique clusters")

    # ─── Step 4: ITS validation ───────────────────────────────────────────────
    print("\n=== Step 4: ITS validation ===")
    LAGS_FRAMES = [1, 2, 5, 10, 20, 50, 100, 200]
    LAGS_NS     = [l * DT_NS for l in LAGS_FRAMES]
    NITS = 4

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (sys_name, cfg) in zip(axes, SYSTEMS.items()):
        print(f"\n  {sys_name}...")
        dtrajs = system_dtrajs[sys_name]
        timescales = {i: [] for i in range(NITS)}
        valid_lags = []

        for lag in LAGS_FRAMES:
            try:
                M = msm.estimate_markov_model(dtrajs, lag=lag, reversible=True)
                ts = M.timescales()[:NITS] * DT_NS
                for i in range(NITS):
                    timescales[i].append(ts[i] if i < len(ts) else np.nan)
                valid_lags.append(lag * DT_NS)
                print(f"    lag={lag*DT_NS:.0f}ns: ts1={ts[0]:.1f}ns")
            except Exception as e:
                print(f"    lag={lag} failed: {e}")

        colors_ts = ['blue', 'red', 'green', 'orange']
        for i in range(NITS):
            ts_arr = np.array(timescales[i])
            valid = ~np.isnan(ts_arr)
            if valid.any():
                ax.semilogy(np.array(valid_lags)[valid], ts_arr[valid],
                           'o-', color=colors_ts[i], lw=1.5, ms=5,
                           label=f'TS {i+1}')

        ax.axvline(10*DT_NS, color='red', ls='--', lw=1, label='lag=5ns')
        ax.set_xlabel('Lag time (ns)')
        ax.set_ylabel('Implied timescale (ns)')
        ax.set_title(sys_name)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle('Implied timescales — strided trajectories (500 ps/frame)')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'its_strided.png'), dpi=150)
    plt.close()
    print("  -> its_strided.png saved")

    # ─── Step 5: MSM + PCCA ───────────────────────────────────────────────────
    print("\n=== Step 5: MSM + PCCA ===")
    MSM_LAG = 10  # 5 ns — adjust after seeing ITS
    N_MACRO = 4

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    populations = {}
    timescale_summary = {}

    for ax, (sys_name, cfg) in zip(axes, SYSTEMS.items()):
        print(f"\n  {sys_name}...")
        dtrajs = system_dtrajs[sys_name]
        M = msm.estimate_markov_model(dtrajs, lag=MSM_LAG, reversible=True)
        print(f"    Active states: {M.nstates}")
        ts = M.timescales()[:5] * DT_NS
        print(f"    Top 5 timescales (ns): {ts}")
        M.pcca(N_MACRO)

        tica_cat = system_tica[sys_name]
        pyemma.plots.plot_free_energy(tica_cat[:,0], tica_cat[:,1], ax=ax, cbar=False)

        pops = []
        macro_colors = ['red', 'lime', 'blue', 'orange']
        centers = kmeans.cluster_centers_
        for i in range(N_MACRO):
            macro_idx = M.metastable_sets[i]
            pop = M.pi[macro_idx].sum()
            pops.append(pop)
            mc = centers[macro_idx]
            ax.scatter(mc[:,0], mc[:,1], c=macro_colors[i], s=30, zorder=5, alpha=0.7)
            ax.annotate(f'S{i}\n{pop:.2f}', (mc[:,0].mean(), mc[:,1].mean()),
                       fontsize=8, ha='center', color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2',
                                facecolor=macro_colors[i], alpha=0.7))

        populations[sys_name] = pops
        timescale_summary[sys_name] = ts
        ax.set_title(sys_name)
        ax.set_xlabel('Joint TICA IC1')
        ax.set_ylabel('Joint TICA IC2' if ax == axes[0] else '')

    plt.suptitle(f'Macrostates — strided MSM (lag={MSM_LAG*DT_NS:.0f} ns)')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'macrostates_strided.png'), dpi=150)
    plt.close()
    print("  -> macrostates_strided.png saved")

    # Population bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(N_MACRO)
    width = 0.25
    for i, (sys_name, cfg) in enumerate(SYSTEMS.items()):
        ax.bar(x + i*width, populations[sys_name], width,
               label=sys_name, color=cfg['color'], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'State {i}' for i in range(N_MACRO)])
    ax.set_ylabel('Equilibrium population')
    ax.set_title('Macrostate populations — strided MSM')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'populations_strided.png'), dpi=150)
    plt.close()
    print("  -> populations_strided.png saved")

    # Summary
    with open(os.path.join(BASE_OUT, 'msm_strided_summary.txt'), 'w') as f:
        f.write(f"=== Strided MSM Summary (500 ps/frame) ===\n")
        f.write(f"MSM lag: {MSM_LAG} frames ({MSM_LAG*DT_NS:.0f} ns)\n\n")
        for sys_name in SYSTEMS:
            f.write(f"{sys_name}:\n")
            f.write(f"  Top 5 timescales (ns): {timescale_summary[sys_name]}\n")
            for i, pop in enumerate(populations[sys_name]):
                f.write(f"  Macrostate {i}: {pop:.3f}\n")
            f.write("\n")

    print(f"\n=== Done. Results in {BASE_OUT}/ ===")
