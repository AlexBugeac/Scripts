"""
save_tica_model.py
Rerun TICA on the same data and save the fitted model for projection
"""

import os
import numpy as np
import mdtraj as md
import pyemma
import pyemma.coordinates as coor
import warnings
warnings.filterwarnings('ignore')

BASE   = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/salieri-data'
TOP    = '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-native/8RJJ-native_05_minimized.pdb'
OUTDIR = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_output_full'

TRAJS = []
for rep in ['replica_01', 'replica_02', 'replica_03']:
    for traj in ['traj.dcd', 'traj2.dcd', 'traj3.dcd']:
        path = os.path.join(BASE, rep, traj)
        if os.path.exists(path):
            TRAJS.append(path)

if __name__ == '__main__':

    # Same featurization as the full pipeline
    print("Building featurizer...")
    feat = coor.featurizer(TOP)
    feat.add_backbone_torsions(periodic=False)
    print(f"  Features: {feat.dimension()}")

    print("Loading data...")
    data = coor.load(TRAJS, features=feat, chunksize=10000, n_jobs=4)

    print("Fitting TICA...")
    tica = coor.tica(data, lag=10, dim=10, kinetic_map=True, n_jobs=1)

    # Save the fitted TICA model
    tica_model_path = os.path.join(OUTDIR, 'tica_model.pyemma')
    tica.save(tica_model_path, overwrite=True)
    print(f"TICA model saved to {tica_model_path}")

    # Also save the featurizer separately
    feat.save(os.path.join(OUTDIR, 'featurizer.pyemma'), overwrite=True)
    print("Featurizer saved.")
