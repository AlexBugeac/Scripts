"""
msm_joint_normalized.py
Joint TICA with per-system feature normalization before concatenation
Prevents any single system from dominating the shared TICA space
"""

import os
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_OUT = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_joint_norm'
os.makedirs(BASE_OUT, exist_ok=True)

HAIRPIN_CORE = list(range(8, 20))
DT_NS = 0.5

SYSTEMS = {
    '8RJJ-native': {
        'top':   '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-native/8RJJ-native_05_minimized.pdb',
        'trajs': [f'/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/salieri-data/replica_0{i}_s50_combined.dcd' for i in [1,2,3]],
        'color': '#E74C3C',
        'label': '8RJJ-native',
    },
    '8RJJ-SS411-424': {
        'top':   '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-m_I411C_S424C_db411-424/8RJJ-m_I411C_S424C_db411-424_05_minimized.pdb',
        'trajs': [f'/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/albicastro-data/8RJJ-m_I411C_S424C_db411-424/replica_0{i}_s50_combined.dcd' for i in [1,2,3]],
        'color': '#2ECC71',
        'label': '8RJJ-SS(411-424)',
    },
    '8RJJ-SS424-429': {
        'top':   '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-m_I411C_S424C_db424-429/8RJJ-m_I411C_S424C_db424-429_05_minimized.pdb',
        'trajs': [f'/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/beethoven-data/8RJJ-m_I411C_S424C_db424-429/replica_0{i}_s50_combined.dcd' for i in [1,2,3]],
        'color': '#3498DB',
        'label': '8RJJ-SS(424-429)',
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

    # ─── Step 1: Featurize each system and normalize ───────────────────────────
    print("=== Step 1: Featurize + normalize per system ===")
    system_data_raw = {}
    system_lengths  = {}

    for sys_name, cfg in SYSTEMS.items():
        print(f"\n  {sys_name}...")
        trajs = [t for t in cfg['trajs'] if os.path.exists(t)]
        feat  = build_feat(cfg['top'])
        data  = coor.load(trajs, features=feat, chunksize=5000, n_jobs=1)
        n     = sum(len(d) for d in data)
        print(f"    Frames: {n}, Features: {feat.dimension()}")

        # Normalize: compute mean and std across all frames, apply z-score
        all_frames = np.concatenate(data)
        mean = all_frames.mean(axis=0)
        std  = all_frames.std(axis=0)
        std[std < 1e-8] = 1.0  # avoid division by zero

        data_norm = [(d - mean) / std for d in data]
        system_data_raw[sys_name] = data_norm
        system_lengths[sys_name]  = [len(d) for d in data_norm]
        print(f"    Normalized. Feature mean range: [{mean.min():.2f}, {mean.max():.2f}]")
        del feat, all_frames

    # ─── Step 2: Joint TICA on normalized data ────────────────────────────────
    print("\n=== Step 2: Joint TICA on normalized features ===")
    all_data = []
    for sys_name in SYSTEMS:
        all_data.extend(system_data_raw[sys_name])

    print(f"  Total datasets: {len(all_data)}")
    print(f"  Total frames: {sum(len(d) for d in all_data)}")

    tica = coor.tica(all_data, lag=10, dim=5, kinetic_map=True, n_jobs=1)
    print(f"  Cumvar (top 3): {tica.cumvar[:3]}")
    tica.save(os.path.join(BASE_OUT, 'joint_tica_norm.pyemma'), overwrite=True)

    # Project each system
    system_tica = {}
    idx = 0
    for sys_name, cfg in SYSTEMS.items():
        n_trajs  = len(system_lengths[sys_name])
        tica_cat = np.concatenate(tica.transform(all_data[idx:idx+n_trajs]))
        system_tica[sys_name] = tica_cat
        np.save(os.path.join(BASE_OUT, f'tica_{sys_name}.npy'), tica_cat)
        print(f"  {sys_name}: {tica_cat.shape[0]} frames, "
              f"IC1=[{tica_cat[:,0].min():.2f},{tica_cat[:,0].max():.2f}], "
              f"IC2=[{tica_cat[:,1].min():.2f},{tica_cat[:,1].max():.2f}]")
        idx += n_trajs

    # ─── Step 3: Individual landscapes ────────────────────────────────────────
    print("\n=== Step 3: Landscapes ===")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    for ax, (sys_name, cfg) in zip(axes, SYSTEMS.items()):
        tica_cat = system_tica[sys_name]
        pyemma.plots.plot_free_energy(tica_cat[:,0], tica_cat[:,1], ax=ax)
        ax.set_title(f"{cfg['label']}\n({tica_cat.shape[0]*DT_NS/1000:.1f} µs)")
        ax.set_xlabel('Joint TICA IC1 (normalized)')
        ax.set_ylabel('Joint TICA IC2' if ax == axes[0] else '')
    plt.suptitle('Free energy landscapes — joint TICA with feature normalization', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'joint_landscapes_norm.png'), dpi=150)
    plt.close()
    print("  -> joint_landscapes_norm.png saved")

    # ─── Step 4: Joint clustering ─────────────────────────────────────────────
    print("\n=== Step 4: Joint clustering (k=100) ===")
    from sklearn.cluster import MiniBatchKMeans
    all_tica = np.concatenate(list(system_tica.values()))
    kmeans   = MiniBatchKMeans(n_clusters=100, random_state=42, n_init=10)
    kmeans.fit(all_tica[:, :2])
    centers  = kmeans.cluster_centers_

    system_dtrajs = {}
    for sys_name, tica_cat in system_tica.items():
        labels = kmeans.predict(tica_cat[:, :2])
        lengths = system_lengths[sys_name]
        dtrajs = []
        start = 0
        for l in lengths:
            dtrajs.append(labels[start:start+l])
            start += l
        system_dtrajs[sys_name] = dtrajs
        print(f"  {sys_name}: {len(np.unique(labels))} unique clusters")

    # ─── Step 5: MSM + PCCA per system ────────────────────────────────────────
    print("\n=== Step 5: MSM + PCCA (lag=20 frames = 10 ns) ===")
    MSM_LAG = 20
    N_MACRO = 4
    results = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    macro_colors = ['red', 'lime', 'blue', 'orange']

    for ax, (sys_name, cfg) in zip(axes, SYSTEMS.items()):
        print(f"\n  {sys_name}...")
        dtrajs = system_dtrajs[sys_name]
        M = msm.estimate_markov_model(dtrajs, lag=MSM_LAG, reversible=True)
        M.pcca(N_MACRO)
        print(f"    Active states: {M.nstates}")
        print(f"    Timescales (ns): {M.timescales()[:4]*DT_NS}")

        pops = []
        for i in range(N_MACRO):
            pop = M.pi[M.metastable_sets[i]].sum()
            pops.append(pop)
            print(f"    Macrostate {i}: {pop:.3f}")

        results[sys_name] = {'pops': pops, 'ts': M.timescales()[:4]*DT_NS}

        tica_cat = system_tica[sys_name]
        pyemma.plots.plot_free_energy(tica_cat[:,0], tica_cat[:,1], ax=ax, cbar=False)
        for i in range(N_MACRO):
            mc  = centers[M.metastable_sets[i]]
            pop = pops[i]
            ax.scatter(mc[:,0], mc[:,1], c=macro_colors[i], s=30, zorder=5, alpha=0.7)
            ax.annotate(f'S{i}\n{pop:.2f}', (mc[:,0].mean(), mc[:,1].mean()),
                       fontsize=8, ha='center', color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2',
                                facecolor=macro_colors[i], alpha=0.7))
        ax.set_title(cfg['label'])
        ax.set_xlabel('Joint TICA IC1')
        ax.set_ylabel('Joint TICA IC2' if ax == axes[0] else '')

    plt.suptitle(f'Macrostates — normalized joint TICA (lag={MSM_LAG*DT_NS:.0f} ns)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'joint_macrostates_norm.png'), dpi=150)
    plt.close()
    print("  -> joint_macrostates_norm.png saved")

    # ─── Step 6: Population bar chart ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(N_MACRO)
    width = 0.25
    for i, (sys_name, cfg) in enumerate(SYSTEMS.items()):
        ax.bar(x + i*width, results[sys_name]['pops'], width,
               label=cfg['label'], color=cfg['color'], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'State {i}' for i in range(N_MACRO)])
    ax.set_ylabel('Equilibrium population')
    ax.set_title('Macrostate populations — normalized joint TICA (lag=10 ns)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'populations_norm.png'), dpi=150)
    plt.close()
    print("  -> populations_norm.png saved")

    # ─── Summary ──────────────────────────────────────────────────────────────
    with open(os.path.join(BASE_OUT, 'summary.txt'), 'w') as f:
        f.write("=== Normalized Joint TICA MSM ===\n")
        f.write(f"Normalization: z-score per system before joint TICA\n")
        f.write(f"MSM lag: {MSM_LAG} frames ({MSM_LAG*DT_NS:.0f} ns)\n\n")
        for sys_name in SYSTEMS:
            f.write(f"{sys_name}:\n")
            f.write(f"  Timescales (ns): {results[sys_name]['ts']}\n")
            for i, pop in enumerate(results[sys_name]['pops']):
                f.write(f"  Macrostate {i}: {pop:.3f}\n")
            f.write("\n")

    print(f"\n=== Done. Results in {BASE_OUT}/ ===")
