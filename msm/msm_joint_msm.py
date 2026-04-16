"""
msm_joint_msm.py
Build MSMs for each system using joint TICA coordinates
Macrostates are directly comparable across systems
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_OUT = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_joint'

SYSTEMS = {
    '8RJJ-native':   {'color': '#E74C3C', 'label': '8RJJ-native'},
    '8RJJ-SS411-424':{'color': '#2ECC71', 'label': '8RJJ-SS(411-424)'},
    '8RJJ-SS424-429':{'color': '#3498DB', 'label': '8RJJ-SS(424-429)'},
}

N_MACRO  = 4
MSM_LAG  = 50
N_CLUST  = 100

if __name__ == '__main__':
    import pyemma
    import pyemma.coordinates as coor
    import pyemma.msm as msm

    # ─── Load joint TICA output ────────────────────────────────────────────────
    print("Loading joint TICA outputs...")
    system_tica = {}
    for sys_name in SYSTEMS:
        path = os.path.join(BASE_OUT, f'tica_{sys_name}.npy')
        system_tica[sys_name] = np.load(path)
        print(f"  {sys_name}: {system_tica[sys_name].shape[0]} frames")

    # ─── Cluster in joint TICA space ──────────────────────────────────────────
    # First cluster on ALL data combined so cluster centers are shared
    print(f"\nClustering all systems jointly (k={N_CLUST})...")
    all_tica = np.concatenate(list(system_tica.values()))
    print(f"  Total frames for clustering: {len(all_tica)}")

    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=N_CLUST, random_state=42, n_init=10)
    kmeans.fit(all_tica[:, :2])  # cluster on IC1, IC2
    centers = kmeans.cluster_centers_
    print(f"  Cluster centers shape: {centers.shape}")

    # Assign each system's frames to clusters
    system_dtrajs = {}
    for sys_name, tica_cat in system_tica.items():
        labels = kmeans.predict(tica_cat[:, :2])
        system_dtrajs[sys_name] = labels
        print(f"  {sys_name}: assigned to {len(np.unique(labels))} unique clusters")

    # ─── Build MSM per system using shared clustering ──────────────────────────
    print(f"\nBuilding MSMs (lag={MSM_LAG} frames = {MSM_LAG*10} ps)...")
    system_msm = {}
    summaries = {}

    for sys_name in SYSTEMS:
        print(f"\n  {sys_name}...")
        # Need list of dtrajs per trajectory — use full dtraj as single sequence
        # (approximation — ideally split by replica but works for population estimates)
        dtrajs = [system_dtrajs[sys_name]]
        try:
            M = msm.estimate_markov_model(dtrajs, lag=MSM_LAG, reversible=True)
            print(f"    Active states: {M.nstates}")
            print(f"    Top 5 timescales (ns): {M.timescales()[:5]*10/1000}")
            M.pcca(N_MACRO)
            system_msm[sys_name] = M

            pops = []
            for i in range(N_MACRO):
                pop = M.pi[M.metastable_sets[i]].sum()
                pops.append(pop)
                print(f"    Macrostate {i}: {pop:.3f}")

            summaries[sys_name] = {
                'timescales': M.timescales()[:5]*10/1000,
                'populations': pops,
                'nstates': M.nstates,
            }
        except Exception as e:
            print(f"    MSM failed: {e}")
            system_msm[sys_name] = None

    # ─── Plot macrostates on joint landscape ──────────────────────────────────
    print("\nPlotting macrostates on joint landscapes...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    for ax, (sys_name, cfg) in zip(axes, SYSTEMS.items()):
        tica_cat = system_tica[sys_name]
        pyemma.plots.plot_free_energy(tica_cat[:,0], tica_cat[:,1], ax=ax, cbar=False)

        M = system_msm.get(sys_name)
        if M is not None:
            macro_colors = ['red', 'lime', 'blue', 'orange']
            for i in range(N_MACRO):
                macro_centers = centers[M.metastable_sets[i]]
                pop = M.pi[M.metastable_sets[i]].sum()
                ax.scatter(macro_centers[:,0], macro_centers[:,1],
                          c=macro_colors[i], s=40, zorder=5, alpha=0.7)
                # Label at centroid
                cx = macro_centers[:,0].mean()
                cy = macro_centers[:,1].mean()
                ax.annotate(f'S{i}\n{pop:.2f}', (cx, cy),
                           fontsize=8, ha='center', color='white',
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2',
                                    facecolor=macro_colors[i], alpha=0.7))

        ax.set_title(f"{cfg['label']}")
        ax.set_xlabel('Joint TICA IC1')
        ax.set_ylabel('Joint TICA IC2' if ax == axes[0] else '')

    plt.suptitle(f'Macrostates on joint TICA landscape (lag={MSM_LAG*10} ps)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'joint_macrostates.png'), dpi=150)
    plt.close()
    print("  -> joint_macrostates.png saved")

    # ─── Population comparison bar chart ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(N_MACRO)
    width = 0.25
    macro_colors = ['red', 'lime', 'blue', 'orange']

    for i, (sys_name, cfg) in enumerate(SYSTEMS.items()):
        if sys_name in summaries:
            pops = summaries[sys_name]['populations']
            bars = ax.bar(x + i*width, pops, width,
                         label=cfg['label'], color=cfg['color'], alpha=0.8)

    ax.set_xlabel('Macrostate')
    ax.set_ylabel('Equilibrium population')
    ax.set_title('Macrostate populations across systems\n(joint TICA + shared clustering)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'State {i}' for i in range(N_MACRO)])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'macrostate_populations.png'), dpi=150)
    plt.close()
    print("  -> macrostate_populations.png saved")

    # ─── Timescale comparison ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for sys_name, cfg in SYSTEMS.items():
        if sys_name in summaries:
            ts = summaries[sys_name]['timescales']
            ax.plot(range(1, len(ts)+1), ts, 'o-',
                   color=cfg['color'], label=cfg['label'], lw=2, ms=8)

    ax.set_xlabel('Process index')
    ax.set_ylabel('Implied timescale (ns)')
    ax.set_title('Implied timescales comparison')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'timescale_comparison.png'), dpi=150)
    plt.close()
    print("  -> timescale_comparison.png saved")

    # ─── Summary ──────────────────────────────────────────────────────────────
    with open(os.path.join(BASE_OUT, 'joint_msm_summary.txt'), 'w') as f:
        f.write("=== Joint MSM Summary ===\n")
        f.write(f"Clustering: k={N_CLUST} on joint TICA IC1+IC2\n")
        f.write(f"MSM lag: {MSM_LAG} frames ({MSM_LAG*10} ps)\n")
        f.write(f"Macrostates: {N_MACRO}\n\n")
        for sys_name, s in summaries.items():
            f.write(f"{sys_name}:\n")
            f.write(f"  Active states: {s['nstates']}\n")
            f.write(f"  Top 5 timescales (ns): {s['timescales']}\n")
            for i, pop in enumerate(s['populations']):
                f.write(f"  Macrostate {i}: {pop:.3f}\n")
            f.write("\n")

    print(f"\n=== Done. Results in {BASE_OUT}/ ===")
