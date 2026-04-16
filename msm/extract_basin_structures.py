"""
extract_basin_structures.py
Extract representative structures from the lowest energy basins
"""

import os
import numpy as np
import mdtraj as md
import pyemma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUTDIR   = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_output_full'
BASE     = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/salieri-data'
TOP      = '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-native/8RJJ-native_05_minimized.pdb'
STRUCT_DIR = os.path.join(OUTDIR, 'basin_structures')
os.makedirs(STRUCT_DIR, exist_ok=True)

# Load TICA model and output
tica     = pyemma.load(os.path.join(OUTDIR, 'tica_model.pyemma'))
feat     = pyemma.load(os.path.join(OUTDIR, 'featurizer.pyemma'))
tica_cat = np.load(os.path.join(OUTDIR, 'tica_output.npy'))

# Define basin centers based on the landscape
# Adjust these coordinates based on what you see in the plot
basins = {
    'top_left_1':  (-0.85,  1.05),   # top-left upper basin
    'top_left_2':  (-0.50,  0.55),   # top-left lower basin
    'bottom_1':    (-0.35, -0.85),   # bottom upper basin
    'bottom_2':    (-0.25, -1.45),   # bottom lower basin
    'right_1':     ( 0.85,  0.45),   # right upper basin
    'right_2':     ( 1.45,  0.35),   # right lower basin
}

print("Finding closest frames to each basin center...")
for basin_name, (ic1_target, ic2_target) in basins.items():
    # Euclidean distance in TICA space
    dist = np.sqrt((tica_cat[:, 0] - ic1_target)**2 +
                   (tica_cat[:, 1] - ic2_target)**2)
    closest_frame_global = np.argmin(dist)
    print(f"\n  Basin {basin_name}: target ({ic1_target}, {ic2_target})")
    print(f"    Closest frame: {closest_frame_global}")
    print(f"    TICA coords:   ({tica_cat[closest_frame_global, 0]:.3f}, {tica_cat[closest_frame_global, 1]:.3f})")
    print(f"    Distance:      {dist[closest_frame_global]:.3f}")

    # Map global frame index to replica/traj/local frame
    TRAJS = []
    for rep in ['replica_01', 'replica_02', 'replica_03']:
        for traj in ['traj.dcd', 'traj2.dcd', 'traj3.dcd']:
            path = os.path.join(BASE, rep, traj)
            if os.path.exists(path):
                TRAJS.append(path)

    # Find which trajectory and local frame
    cumulative = 0
    for traj_path in TRAJS:
        t = md.load(traj_path, top=TOP)
        n = t.n_frames
        if closest_frame_global < cumulative + n:
            local_frame = closest_frame_global - cumulative
            frame = t[local_frame]
            out_pdb = os.path.join(STRUCT_DIR, f'basin_{basin_name}.pdb')
            frame.save_pdb(out_pdb)
            print(f"    Saved to: {out_pdb}")
            break
        cumulative += n

print("\nDone. Basin structures saved to:", STRUCT_DIR)

# Plot with basin markers
fig, ax = plt.subplots(figsize=(9, 7))
pyemma.plots.plot_free_energy(tica_cat[:, 0], tica_cat[:, 1], ax=ax)
colors = ['lime', 'green', 'cyan', 'blue', 'orange', 'red']
for (name, (ic1, ic2)), color in zip(basins.items(), colors):
    ax.scatter(ic1, ic2, c=color, s=150, zorder=10, marker='X',
               edgecolors='black', linewidths=0.5, label=name)
ax.set_xlabel('TICA IC1')
ax.set_ylabel('TICA IC2')
ax.set_title('Basin representative structures')
ax.legend(fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'basin_locations.png'), dpi=150)
plt.close()
print("-> basin_locations.png saved")
