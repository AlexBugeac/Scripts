import os
import numpy as np
import mdtraj as md
import pyemma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_OUT = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_joint'
TOP_8RJJ = '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-native/8RJJ-native_05_minimized.pdb'
TOP_8W0Y = '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8W0Y-native/8W0Y-native_05_minimized.pdb'
HAIRPIN_CORE = list(range(8, 20))

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
    tica = pyemma.load(os.path.join(BASE_OUT, 'joint_tica_model.pyemma'))
    tica_native = np.load(os.path.join(BASE_OUT, 'tica_8RJJ-native.npy'))

    # Project crystal structures
    feat_rjj = build_feat(TOP_8RJJ)
    t_rjj = md.load(TOP_8RJJ)
    tica_rjj = tica.transform(feat_rjj.transform(t_rjj))
    print(f"8RJJ crystal: IC1={tica_rjj[0,0]:.3f}, IC2={tica_rjj[0,1]:.3f}")

    feat_w0y = build_feat(TOP_8W0Y)
    t_w0y = md.load(TOP_8W0Y)
    tica_w0y = tica.transform(feat_w0y.transform(t_w0y))
    print(f"8W0Y crystal: IC1={tica_w0y[0,0]:.3f}, IC2={tica_w0y[0,1]:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    pyemma.plots.plot_free_energy(tica_native[:,0], tica_native[:,1], ax=ax)
    ax.scatter(tica_rjj[0,0], tica_rjj[0,1], c='lime', s=300, zorder=10,
               marker='*', edgecolors='black', lw=0.5, label='8RJJ crystal')
    ax.scatter(tica_w0y[0,0], tica_w0y[0,1], c='red', s=300, zorder=10,
               marker='*', edgecolors='black', lw=0.5, label='8W0Y crystal')
    ax.set_xlabel('Joint TICA IC1')
    ax.set_ylabel('Joint TICA IC2')
    ax.set_title('Native FES with crystal structures projected\n(joint TICA coordinates)')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUT, 'native_with_crystals.png'), dpi=150)
    plt.close()
    print("-> native_with_crystals.png saved")
