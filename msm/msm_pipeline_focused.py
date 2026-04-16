"""
import resource
resource.setrlimit(resource.RLIMIT_AS, (20 * 1024**3, 20 * 1024**3))
msm_pipeline_focused.py
Focused MSM analysis on AS412 beta-hairpin region (residues 8-19, resSeq 9-20)
Much faster than global featurization — runs in minutes not hours.

Usage:
  python msm_pipeline_focused.py
"""

import os
import numpy as np
import mdtraj as md
import pyemma
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE   = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/salieri-data'
TOP    = '/home/alexb/PhD/cantavac/hcv-e2/simulations/amber/final/8RJJ-native/8RJJ-native_05_minimized.pdb'
OUTDIR = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent/msm_output_focused'
os.makedirs(OUTDIR, exist_ok=True)

TRAJS = []
for rep in ['replica_01', 'replica_02', 'replica_03']:
    for traj in ['traj.dcd', 'traj2.dcd', 'traj3.dcd']:
        path = os.path.join(BASE, rep, traj)
        if os.path.exists(path):
            TRAJS.append(path)

print(f"Total trajectory files: {len(TRAJS)}")

# ─── Define hairpin region ────────────────────────────────────────────────────
# AS412 beta-hairpin: residues 8-19 (MDTraj 0-based index)
# Plus flanking residues 5-22 for context
HAIRPIN_CORE    = list(range(8, 20))    # 8-19 inclusive
HAIRPIN_FLANKED = list(range(5, 23))    # 5-22 inclusive

print(f"Hairpin core residues (0-based): {HAIRPIN_CORE}")
print(f"Flanked region residues (0-based): {HAIRPIN_FLANKED}")

# ─── Featurization ────────────────────────────────────────────────────────────
print("\n=== Featurizing (focused on hairpin) ===")
feat = coor.featurizer(TOP)

# 1. Backbone torsions for flanked hairpin region
feat.add_backbone_torsions(selstr='resid 5 to 22', periodic=False)

# 2. All Ca-Ca pairwise distances within hairpin core
top_mdtraj = md.load_topology(TOP)
ca_pairs = []
hairpin_ca = [a.index for a in top_mdtraj.atoms
              if a.name == 'CA' and a.residue.index in HAIRPIN_CORE]
for i in range(len(hairpin_ca)):
    for j in range(i+2, len(hairpin_ca)):  # skip i+1 (bonded)
        ca_pairs.append([hairpin_ca[i], hairpin_ca[j]])
feat.add_distances(ca_pairs)

# 3. Key long-range contacts: hairpin vs rest of protein
#    ILE11(idx 10) - HIS18(idx 17) backbone contacts from H-bond analysis
#    THR13(idx 12) - SER16(idx 15)
key_pairs_resid = [(10, 17), (12, 15), (11, 18), (9, 18)]
key_ca_pairs = []
for r1, r2 in key_pairs_resid:
    a1 = [a.index for a in top_mdtraj.atoms if a.name == 'CA' and a.residue.index == r1][0]
    a2 = [a.index for a in top_mdtraj.atoms if a.name == 'CA' and a.residue.index == r2][0]
    key_ca_pairs.append([a1, a2])
feat.add_distances(key_ca_pairs)

print(f"  Total features: {feat.dimension()}")
print(f"  Backbone torsions (res 5-22) + Ca distances (hairpin) + key contacts")

# ─── Load data ────────────────────────────────────────────────────────────────
print("\nLoading trajectories...")
data = coor.load(TRAJS, features=feat, chunksize=10000, n_jobs=1)
print(f"  Loaded {len(data)} trajectories")
print(f"  Lengths: {[len(d) for d in data]}")

# ─── TICA ─────────────────────────────────────────────────────────────────────
print("\n=== TICA (lag=10 frames = 100 ps) ===")
tica = coor.tica(data, lag=10, dim=5, kinetic_map=True, n_jobs=1)
tica_output = tica.get_output()
tica_cat = np.concatenate(tica_output)
print(f"  TICA dimensions: {tica.dimension()}")
print(f"  Cumulative kinetic variance: {tica.cumvar}")
np.save(os.path.join(OUTDIR, 'tica_output.npy'), tica_cat)

# Free energy landscape
fig, ax = plt.subplots(figsize=(8, 6))
pyemma.plots.plot_free_energy(tica_cat[:, 0], tica_cat[:, 1], ax=ax)
ax.set_xlabel('TICA IC1 (slowest motion)')
ax.set_ylabel('TICA IC2')
ax.set_title('Free energy landscape — AS412 hairpin (IC1 vs IC2)')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'tica_landscape.png'), dpi=150)
plt.close()
print("  -> tica_landscape.png saved")

# ─── Clustering ───────────────────────────────────────────────────────────────
print("\n=== Clustering (k-means, k=100) ===")
cluster = coor.cluster_kmeans(tica_output, k=100, max_iter=100, stride=10, n_jobs=1)
dtrajs  = cluster.dtrajs
print(f"  Cluster centers: {cluster.n_clusters}")

# ─── Implied timescales ───────────────────────────────────────────────────────
print("\n=== Implied timescales ===")
lags = [1, 2, 5, 10, 20, 50, 100, 200, 500]
its  = msm.its(dtrajs, lags=lags, nits=5, reversible=True)

fig, ax = plt.subplots(figsize=(8, 5))
mplt.plot_implied_timescales(its, ax=ax, units='frames', dt=1)
ax.set_xlabel('Lag time (frames, 1 frame = 10 ps)')
ax.set_ylabel('Implied timescale (frames)')
ax.set_title('Implied timescales — choose lag where lines plateau')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'implied_timescales.png'), dpi=150)
plt.close()
print("  -> implied_timescales.png saved")
print("  IMPORTANT: review this plot before interpreting macrostates")

# ─── MSM ──────────────────────────────────────────────────────────────────────
MSM_LAG = 50
print(f"\n=== MSM (lag={MSM_LAG} frames = {MSM_LAG*10} ps) ===")
M = msm.estimate_markov_model(dtrajs, lag=MSM_LAG, reversible=True)
print(f"  Active states: {M.n_states}")
print(f"  Active count fraction: {M.active_count_fraction:.3f}")
print(f"  Top 5 timescales (ns): {M.timescales()[:5]*10/1000:.2f}")

# ─── PCCA+ macrostates ────────────────────────────────────────────────────────
N_MACRO = 4
print(f"\n=== PCCA+ ({N_MACRO} macrostates) ===")
M.pcca(N_MACRO)

fig, axes = plt.subplots(1, N_MACRO, figsize=(4*N_MACRO, 4))
for i, ax in enumerate(axes):
    pyemma.plots.plot_free_energy(
        tica_cat[:, 0], tica_cat[:, 1], ax=ax, cbar=False
    )
    macro_states = M.metastable_sets[i]
    centers = cluster.clustercenters[macro_states]
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=30, zorder=5)
    pop = M.pi[macro_states].sum()
    ax.set_title(f'Macrostate {i}\npop={pop:.3f}')
    ax.set_xlabel('IC1')
    ax.set_ylabel('IC2' if i == 0 else '')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'macrostates.png'), dpi=150)
plt.close()
print("  -> macrostates.png saved")

# ─── Summary ──────────────────────────────────────────────────────────────────
with open(os.path.join(OUTDIR, 'msm_summary.txt'), 'w') as f:
    f.write("=== Focused MSM Summary (AS412 hairpin) ===\n")
    f.write(f"Hairpin residues: 8-19 (resSeq 9-20)\n")
    f.write(f"Total frames: {sum(len(d) for d in data)}\n")
    f.write(f"Features: {feat.dimension()}\n")
    f.write(f"TICA cumvar: {tica.cumvar}\n")
    f.write(f"MSM lag: {MSM_LAG} frames ({MSM_LAG*10} ps)\n")
    f.write(f"Active states: {M.n_states}\n")
    f.write(f"Top 5 timescales (ns): {M.timescales()[:5]*10/1000}\n")
    f.write(f"Macrostates: {N_MACRO}\n")
    for i in range(N_MACRO):
        pop = M.pi[M.metastable_sets[i]].sum()
        f.write(f"  Macrostate {i}: population {pop:.3f}\n")

print("\n=== Done. Review implied_timescales.png first ===")
