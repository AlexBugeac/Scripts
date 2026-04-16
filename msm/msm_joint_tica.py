"""
msm_joint_tica.py
Joint TICA built from all three systems combined
Then each system projected onto the shared TICA space
Focused featurization: hairpin region residues 5-22
"""

import os
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_OUT = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_joint'
os.makedirs(BASE_OUT, exist_ok=True)

HAIRPIN_CORE = list(range(8, 20))

SYSTEMS = {
    '8RJJ-native': {
        'top':  '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-native/8RJJ-native_05_minimized.pdb',
        'base': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/salieri-data',
        'color': '#E74C3C',
        'label': '8RJJ-native',
    },
    '8RJJ-SS411-424': {
        'top':  '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-m_I411C_S424C_db411-424/8RJJ-m_I411C_S424C_db411-424_05_minimized.pdb',
        'base': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/albicastro-data/8RJJ-m_I411C_S424C_db411-424',
        'color': '#2ECC71',
        'label': '8RJJ-SS(411-424)',
    },
    '8RJJ-SS424-429': {
        'top':  '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-m_I411C_S424C_db424-429/8RJJ-m_I411C_S424C_db424-429_05_minimized.pdb',
        'base': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/beethoven-data/8RJJ-m_I411C_S424C_db424-429',
        'color': '#3498DB',
        'label': '8RJJ-SS(424-429)',
    },
}

def get_trajs(base):
    trajs = []
    for rep in ['replica_01', 'replica_02', 'replica_03']:
        for traj in ['traj.dcd', 'traj2.dcd', 'traj3.dcd']:
            path = os.path.join(base, rep, traj)
            if os.path.exists(path):
                trajs.append(path)
    return trajs

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

    # ─── Step 1: Featurize all systems ────────────────────────────────────────
    print("=== Step 1: Featurizing all systems ===")
    all_data = []
    system_lengths = {}

    for sys_name, cfg in SYSTEMS.items():
        print(f"\n  {sys_name}...")
        trajs = get_trajs(cfg['base'])
        feat = build_feat(cfg['top'])
        print(f"    Trajectories: {len(trajs)}, Features: {feat.dimension()}")
        data = coor.load(trajs, features=feat, chunksize=10000, n_jobs=1)
        n_frames = sum(len(d) for d in data)
        system_lengths[sys_name] = [len(d) for d in data]
        all_data.extend(data)
        print(f"    Frames: {n_frames}")
        del feat

    print(f"\n  Total datasets: {len(all_data)}")
    print(f"  Total frames: {sum(len(d) for d in all_data)}")

    # ─── Step 2: Fit joint TICA on all data combined ───────────────────────────
    print("\n=== Step 2: Fitting joint TICA ===")
    tica = coor.tica(all_data, lag=10, dim=5, kinetic_map=True, n_jobs=1)
    print(f"  TICA dimensions: {tica.dimension()}")
    print(f"  Cumvar (top 3): {tica.cumvar[:3]}")
    tica.save(os.path.join(BASE_OUT, 'joint_tica_model.pyemma'), overwrite=True)
    print("  Joint TICA model saved.")

    # ─── Step 3: Project each system separately ───────────────────────────────
    print("\n=== Step 3: Projecting each system ===")
    system_tica = {}
    idx = 0
    for sys_name, cfg in SYSTEMS.items():
        n_trajs = len(system_lengths[sys_name])
        sys_data = all_data[idx:idx+n_trajs]
        tica_out = tica.transform(sys_data)
        tica_cat = np.concatenate(tica_out)
        system_tica[sys_name] = tica_cat
        np.save(os.path.join(BASE_OUT, f'tica_{sys_name}.npy'), tica_cat)
        print(f"  {sys_name}: {tica_cat.shape[0]} frames projected")
        idx += n_trajs

    # ─── Step 4: Comparative plots on shared axes ──────────────────────────────
    print("\n=== Step 4: Generating plots ===")

    # Individual landscapes on shared color scale
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    for ax, (sys_name, cfg) in zip(axes, SYSTEMS.items()):
        tica_cat = system_tica[sys_name]
        pyemma.plots.plot_free_energy(tica_cat[:,0], tica_cat[:,1], ax=ax)
        ax.set_title(f"{cfg['label']}\n({tica_cat.shape[0]//100000:.1f}M frames)")
        ax.set_xlabel('Joint TICA IC1')
        ax.set_ylabel('Joint TICA IC2' if ax == axes[0] else '')
    plt.suptitle('Free energy landscapes on shared TICA coordinates', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'joint_landscapes.png'), dpi=150)
    plt.close()
    print("  -> joint_landscapes.png saved")

    # Overlay plot: scatter of each system colored differently
    fig, ax = plt.subplots(figsize=(9, 7))
    # First plot native as background free energy
    tica_native = system_tica['8RJJ-native']
    pyemma.plots.plot_free_energy(tica_native[:,0], tica_native[:,1],
                                   ax=ax, alpha=0.6, cbar=True)
    # Overlay density contours for each mutant
    for sys_name, cfg in SYSTEMS.items():
        if sys_name == '8RJJ-native':
            continue
        tica_cat = system_tica[sys_name]
        # Plot as contour
        from scipy.stats import gaussian_kde
        xy = np.vstack([tica_cat[:,0], tica_cat[:,1]])
        # Subsample for speed
        idx_sub = np.random.choice(len(tica_cat), min(50000, len(tica_cat)), replace=False)
        ax.scatter(tica_cat[idx_sub,0], tica_cat[idx_sub,1],
                   c=cfg['color'], alpha=0.05, s=0.5, zorder=2)
        ax.plot([], [], color=cfg['color'], lw=2, label=cfg['label'])

    ax.plot([], [], color='gray', lw=2, label='8RJJ-native (background)')
    ax.set_xlabel('Joint TICA IC1')
    ax.set_ylabel('Joint TICA IC2')
    ax.set_title('System comparison on shared TICA space\n(native=background FES, mutants=scatter)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'joint_overlay.png'), dpi=150)
    plt.close()
    print("  -> joint_overlay.png saved")

    print(f"\n=== Done. Results in {BASE_OUT}/ ===")
