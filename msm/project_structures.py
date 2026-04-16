import os
import numpy as np
import mdtraj as md
import pyemma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUTDIR = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_output_full'
TOP_8RJJ = '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-native/8RJJ-native_05_minimized.pdb'
TOP_8W0Y = '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8W0Y-native/8W0Y-native_05_minimized.pdb'

# Load saved model
tica = pyemma.load(os.path.join(OUTDIR, 'tica_model.pyemma'))
feat = pyemma.load(os.path.join(OUTDIR, 'featurizer.pyemma'))
tica_cat = np.load(os.path.join(OUTDIR, 'tica_output.npy'))

# Project 8RJJ crystal structure
t_8rjj = md.load(TOP_8RJJ)
tica_8rjj = tica.transform(feat.transform(t_8rjj))
print(f"8RJJ: IC1={tica_8rjj[0,0]:.3f}, IC2={tica_8rjj[0,1]:.3f}")

# Project 8W0Y crystal structure
t_8w0y = md.load(TOP_8W0Y)
tica_8w0y = tica.transform(feat.transform(t_8w0y))
print(f"8W0Y: IC1={tica_8w0y[0,0]:.3f}, IC2={tica_8w0y[0,1]:.3f}")

# Plot landscape with crystal structures marked
fig, ax = plt.subplots(figsize=(9, 7))
pyemma.plots.plot_free_energy(tica_cat[:, 0], tica_cat[:, 1], ax=ax)
ax.scatter(tica_8rjj[0, 0], tica_8rjj[0, 1], c='lime', s=200, zorder=10,
           marker='*', label='8RJJ crystal', edgecolors='black', linewidths=0.5)
ax.scatter(tica_8w0y[0, 0], tica_8w0y[0, 1], c='red', s=200, zorder=10,
           marker='*', label='8W0Y crystal', edgecolors='black', linewidths=0.5)
ax.set_xlabel('TICA IC1 (slowest motion)')
ax.set_ylabel('TICA IC2')
ax.set_title('Free energy landscape with crystal structures projected')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'landscape_with_structures.png'), dpi=150)
plt.close()
print("-> landscape_with_structures.png saved")
