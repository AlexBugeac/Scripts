"""
msm_all_systems.py
Builds focused MSM for 8RJJ-native, db411-424, db424-429
Runs sequentially to control RAM usage
Hairpin-focused featurization: backbone torsions res 5-22 + Ca distances + key contacts
"""

import os
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_OUT = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_comparative'
os.makedirs(BASE_OUT, exist_ok=True)

SYSTEMS = {
    '8RJJ-native': {
        'top':  '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-native/8RJJ-native_05_minimized.pdb',
        'base': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/salieri-data',
        'reps': ['replica_01', 'replica_02', 'replica_03'],
        'trajs': ['traj.dcd', 'traj2.dcd', 'traj3.dcd'],
        'color': '#E74C3C',
    },
    '8RJJ-SS411-424': {
        'top':  '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-m_I411C_S424C_db411-424/8RJJ-m_I411C_S424C_db411-424_05_minimized.pdb',
        'base': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/albicastro-data/8RJJ-m_I411C_S424C_db411-424',
        'reps': ['replica_01', 'replica_02', 'replica_03'],
        'trajs': ['traj.dcd', 'traj2.dcd', 'traj3.dcd'],
        'color': '#2ECC71',
    },
    '8RJJ-SS424-429': {
        'top':  '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-m_I411C_S424C_db424-429/8RJJ-m_I411C_S424C_db424-429_05_minimized.pdb',
        'base': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/beethoven-data/8RJJ-m_I411C_S424C_db424-429',
        'reps': ['replica_01', 'replica_02', 'replica_03'],
        'trajs': ['traj.dcd', 'traj2.dcd', 'traj3.dcd'],
        'color': '#3498DB',
    },
}

HAIRPIN_CORE    = list(range(8, 20))
HAIRPIN_FLANKED = list(range(5, 23))

if __name__ == '__main__':
    import pyemma
    import pyemma.coordinates as coor
    import pyemma.msm as msm

    landscapes = {}  # store tica_cat for each system for comparison plot

    for sys_name, cfg in SYSTEMS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {sys_name}")
        print(f"{'='*60}")

        OUTDIR = os.path.join(BASE_OUT, sys_name)
        os.makedirs(OUTDIR, exist_ok=True)

        # Collect trajectories
        TRAJS = []
        for rep in cfg['reps']:
            for traj in cfg['trajs']:
                path = os.path.join(cfg['base'], rep, traj)
                if os.path.exists(path):
                    TRAJS.append(path)
        print(f"  Trajectories: {len(TRAJS)}")

        # Featurization
        print("  Featurizing...")
        feat = coor.featurizer(cfg['top'])
        feat.add_backbone_torsions(selstr='resid 5 to 22', periodic=False)

        top_md = md.load_topology(cfg['top'])
        hairpin_ca = [a.index for a in top_md.atoms
                      if a.name == 'CA' and a.residue.index in HAIRPIN_CORE]
        ca_pairs = []
        for i in range(len(hairpin_ca)):
            for j in range(i+2, len(hairpin_ca)):
                ca_pairs.append([hairpin_ca[i], hairpin_ca[j]])
        feat.add_distances(ca_pairs)

        # Key H-bond contacts
        key_pairs = [(10,17),(12,15),(11,18),(9,18)]
        key_ca = []
        for r1, r2 in key_pairs:
            a1 = [a.index for a in top_md.atoms if a.name=='CA' and a.residue.index==r1]
            a2 = [a.index for a in top_md.atoms if a.name=='CA' and a.residue.index==r2]
            if a1 and a2:
                key_ca.append([a1[0], a2[0]])
        if key_ca:
            feat.add_distances(key_ca)

        print(f"  Features: {feat.dimension()}")

        # Load data
        print("  Loading trajectories (n_jobs=1)...")
        data = coor.load(TRAJS, features=feat, chunksize=10000, n_jobs=1)
        print(f"  Frames: {sum(len(d) for d in data)}")

        # TICA
        print("  Running TICA...")
        tica = coor.tica(data, lag=10, dim=5, kinetic_map=True, n_jobs=1)
        tica_output = tica.get_output()
        tica_cat = np.concatenate(tica_output)
        np.save(os.path.join(OUTDIR, 'tica_output.npy'), tica_cat)
        tica.save(os.path.join(OUTDIR, 'tica_model.pyemma'), overwrite=True)
        print(f"  TICA done. Cumvar: {tica.cumvar[:3]}")

        # Free energy landscape
        fig, ax = plt.subplots(figsize=(8, 6))
        pyemma.plots.plot_free_energy(tica_cat[:,0], tica_cat[:,1], ax=ax)
        ax.set_xlabel('TICA IC1')
        ax.set_ylabel('TICA IC2')
        ax.set_title(f'Free energy landscape — {sys_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'tica_landscape.png'), dpi=150)
        plt.close()

        # Clustering
        print("  Clustering (k=100, n_jobs=1)...")
        cluster = coor.cluster_kmeans(tica_output, k=100, max_iter=100,
                                      stride=10, n_jobs=1)
        dtrajs = cluster.dtrajs
        np.save(os.path.join(OUTDIR, 'dtrajs.npy'), np.array(dtrajs, dtype=object))
        print(f"  Clusters: {cluster.n_clusters}")

        # MSM
        print("  Building MSM (lag=50)...")
        MSM_LAG = 50
        M = msm.estimate_markov_model(dtrajs, lag=MSM_LAG, reversible=True)
        print(f"  Active states: {M.nstates}")
        print(f"  Top 5 timescales (ns): {M.timescales()[:5]*10/1000}")

        # PCCA+
        N_MACRO = 4
        M.pcca(N_MACRO)

        fig, axes = plt.subplots(1, N_MACRO, figsize=(4*N_MACRO, 4))
        for i, ax in enumerate(axes):
            pyemma.plots.plot_free_energy(tica_cat[:,0], tica_cat[:,1],
                                          ax=ax, cbar=False)
            macro_states = M.metastable_sets[i]
            centers = cluster.clustercenters[macro_states]
            ax.scatter(centers[:,0], centers[:,1], c='red', s=30, zorder=5)
            pop = M.pi[macro_states].sum()
            ax.set_title(f'State {i}\npop={pop:.3f}')
            ax.set_xlabel('IC1')
        plt.suptitle(f'{sys_name} — PCCA+ macrostates')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'macrostates.png'), dpi=150)
        plt.close()

        # Summary
        with open(os.path.join(OUTDIR, 'msm_summary.txt'), 'w') as f:
            f.write(f"=== MSM Summary: {sys_name} ===\n")
            f.write(f"Trajectories: {len(TRAJS)}\n")
            f.write(f"Total frames: {sum(len(d) for d in data)}\n")
            f.write(f"Features: {feat.dimension()}\n")
            f.write(f"TICA cumvar (top 3): {tica.cumvar[:3]}\n")
            f.write(f"MSM lag: {MSM_LAG} frames ({MSM_LAG*10} ps)\n")
            f.write(f"Active states: {M.nstates}\n")
            f.write(f"Top 5 timescales (ns): {M.timescales()[:5]*10/1000}\n")
            f.write(f"Macrostates: {N_MACRO}\n")
            for i in range(N_MACRO):
                pop = M.pi[M.metastable_sets[i]].sum()
                f.write(f"  Macrostate {i}: population {pop:.3f}\n")

        landscapes[sys_name] = {
            'tica_cat': tica_cat,
            'color': cfg['color'],
        }

        print(f"  Done. Results in {OUTDIR}/")

        # Free memory before next system
        del data, tica, tica_output, cluster, dtrajs, M
        import gc
        gc.collect()
        print("  Memory freed.")

    # ─── Comparative landscape overlay ────────────────────────────────────────
    print("\nGenerating comparative landscape overlay...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    for ax, (name, d) in zip(axes, landscapes.items()):
        pyemma.plots.plot_free_energy(d['tica_cat'][:,0], d['tica_cat'][:,1], ax=ax)
        ax.set_title(name)
        ax.set_xlabel('TICA IC1')
        ax.set_ylabel('TICA IC2' if ax == axes[0] else '')
    plt.suptitle('Comparative free energy landscapes')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'comparative_landscapes.png'), dpi=150)
    plt.close()
    print(f"Saved: {BASE_OUT}/comparative_landscapes.png")

    print("\n=== All systems complete ===")
