"""
create_strided_dcds.py
Create stride-50 DCDs — memory efficient version
Strides each trajectory separately before joining
"""

import os
import mdtraj as md

STRIDE = 50

SYSTEMS = {
    '8RJJ-native': {
        'top':  '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-native/8RJJ-native_05_minimized.pdb',
        'base': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/salieri-data',
        'trajs': ['traj.dcd', 'traj2.dcd', 'traj3.dcd'],
        'outdir': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/salieri-data',
    },
    '8RJJ-SS411-424': {
        'top':  '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-m_I411C_S424C_db411-424/8RJJ-m_I411C_S424C_db411-424_05_minimized.pdb',
        'base': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/albicastro-data/8RJJ-m_I411C_S424C_db411-424',
        'trajs': ['traj.dcd', 'traj2.dcd', 'traj3.dcd', 'traj4.dcd'],
        'outdir': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/albicastro-data/8RJJ-m_I411C_S424C_db411-424',
    },
    '8RJJ-SS424-429': {
        'top':  '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-m_I411C_S424C_db424-429/8RJJ-m_I411C_S424C_db424-429_05_minimized.pdb',
        'base': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/beethoven-data/8RJJ-m_I411C_S424C_db424-429',
        'trajs': ['traj.dcd', 'traj2.dcd'],
        'outdir': '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/beethoven-data/8RJJ-m_I411C_S424C_db424-429',
    },
}

for sys_name, cfg in SYSTEMS.items():
    print(f"\n=== {sys_name} ===")
    for rep in ['replica_01', 'replica_02', 'replica_03']:
        rep_dir = os.path.join(cfg['base'], rep)
        out_dcd = os.path.join(cfg['outdir'], f'{rep}_s50_combined.dcd')

        # Stride each traj separately and collect
        strided_chunks = []
        for traj_name in cfg['trajs']:
            path = os.path.join(rep_dir, traj_name)
            if not os.path.exists(path):
                continue
            print(f"  {rep}/{traj_name}...", end=' ', flush=True)
            t = md.load(path, top=cfg['top'])
            s = t[::STRIDE]
            print(f"{t.n_frames} -> {s.n_frames} frames")
            strided_chunks.append(s)
            del t  # free memory immediately

        if not strided_chunks:
            print(f"  No trajectories found for {rep}")
            continue

        combined = md.join(strided_chunks)
        combined.save_dcd(out_dcd)
        print(f"  -> {out_dcd}: {combined.n_frames} frames ({combined.n_frames*500/1e6:.2f} us)")
        del strided_chunks, combined

print("\n=== Done ===")
