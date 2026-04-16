"""
msm_full_all.py
Full backbone torsion MSM for all three systems
All 243 residues, ~482 features, sequential to control RAM
"""

import os
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_OUT = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_full'
os.makedirs(BASE_OUT, exist_ok=True)

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

    for sys_name, cfg in SYSTEMS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {sys_name}")
        print(f"{'='*60}")

        OUTDIR = os.path.join(BASE_OUT, sys_name)
        os.makedirs(OUTDIR, exist_ok=True)

        # Collect trajectories
        TRAJS = []
        for rep in ['replica_01', 'replica_02', 'replica_03']:
            for traj in ['traj.dcd', 'traj2.dcd', 'traj3.dcd']:
                path = os.path.join(cfg['base'], rep, traj)
                if os.path.exists(path):
                    TRAJS.append(path)
        print(f"  Trajectories: {len(TRAJS)}")

        # Full featurization
        print("  Featurizing (full backbone torsions)...")
        feat = coor.featurizer(cfg['top'])
        feat.add_backbone_torsions(periodic=False)
        print(f"  Features: {feat.dimension()}")

        # Load
        print("  Loading (n_jobs=1)...")
        data = coor.load(TRAJS, features=feat, chunksize=10000, n_jobs=1)
        print(f"  Frames: {sum(len(d) for d in data)}")

        # TICA
        print("  TICA (lag=10, dim=10)...")
        tica = coor.tica(data, lag=10, dim=10, kinetic_map=True, n_jobs=1)
        tica_output = tica.get_output()
        tica_cat = np.concatenate(tica_output)
        np.save(os.path.join(OUTDIR, 'tica_output.npy'), tica_cat)
        tica.save(os.path.join(OUTDIR, 'tica_model.pyemma'), overwrite=True)
        print(f"  Cumvar (top 3): {tica.cumvar[:3]}")

        # Landscape
        fig, ax = plt.subplots(figsize=(8, 6))
        pyemma.plots.plot_free_energy(tica_cat[:,0], tica_cat[:,1], ax=ax)
        ax.set_xlabel('TICA IC1')
        ax.set_ylabel('TICA IC2')
        ax.set_title(f'Full MSM landscape — {sys_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'tica_landscape.png'), dpi=150)
        plt.close()

        # Clustering
        print("  Clustering (k=200, n_jobs=1)...")
        cluster = coor.cluster_kmeans(tica_output, k=200, max_iter=100,
                                      stride=10, n_jobs=1)
        dtrajs = cluster.dtrajs
        np.save(os.path.join(OUTDIR, 'dtrajs.npy'), np.array(dtrajs, dtype=object))

        # MSM
        print("  MSM (lag=50)...")
        MSM_LAG = 50
        M = msm.estimate_markov_model(dtrajs, lag=MSM_LAG, reversible=True)
        print(f"  Active states: {M.nstates}")
        print(f"  Top 5 timescales (ns): {M.timescales()[:5]*10/1000}")

        # PCCA+
        N_MACRO = 4
        M.pcca(N_MACRO)

        with open(os.path.join(OUTDIR, 'msm_summary.txt'), 'w') as f:
            f.write(f"=== Full MSM: {sys_name} ===\n")
            f.write(f"Features: {feat.dimension()}\n")
            f.write(f"Frames: {sum(len(d) for d in data)}\n")
            f.write(f"TICA cumvar (top 5): {tica.cumvar[:5]}\n")
            f.write(f"MSM lag: {MSM_LAG} frames ({MSM_LAG*10} ps)\n")
            f.write(f"Active states: {M.nstates}\n")
            f.write(f"Top 5 timescales (ns): {M.timescales()[:5]*10/1000}\n")
            for i in range(N_MACRO):
                pop = M.pi[M.metastable_sets[i]].sum()
                f.write(f"Macrostate {i}: {pop:.3f}\n")

        print(f"  Done: {OUTDIR}/")

        del data, tica, tica_output, cluster, dtrajs, M
        import gc; gc.collect()
        print("  Memory freed.")

    print("\n=== Full MSM complete ===")
