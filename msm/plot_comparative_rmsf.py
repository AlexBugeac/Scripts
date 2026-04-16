"""
plot_comparative_rmsf.py
Comparative RMSF plot: 8RJJ-native vs db411-424 vs db424-429
Mean ± SD across 3 replicates, with E2 domain annotations
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

BASE = '/home/alexb/PhD/cantavac/hcv-e2/simulations/implicit_solvent'
OUTDIR = os.path.join(BASE, 'comparative_plots')
os.makedirs(OUTDIR, exist_ok=True)

SEQ_OFFSET = 405  # MDTraj residue 0 = E2 residue 405 (ACE cap, so res 1 = 406... adjust if needed)

# ─── E2 domain definitions ────────────────────────────────────────────────────
E2_DOMAINS = [
    ('AS412',      412, 423, '#E74C3C'),
    ('FL',         423, 459, '#E67E22'),
    ('VR2',        459, 484, '#F1C40F'),
    ('B-san sw 1', 484, 519, '#2ECC71'),
    ('CD81 loop',  519, 536, '#1ABC9C'),
    ('B-san sw 2', 536, 570, '#3498DB'),
    ('VR3',        570, 586, '#9B59B6'),
    ('Post-VR3',   586, 602, '#E91E63'),
    ('BL',         602, 650, '#795548'),
]

def add_domain_bar(ax, x_min, x_max):
    for name, start, end, color in E2_DOMAINS:
        s = max(start, x_min)
        e = min(end, x_max)
        if s >= e:
            continue
        rect = mpatches.FancyBboxPatch(
            (s, -0.12), e - s, 0.06,
            boxstyle="round,pad=0",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text((s + e) / 2, -0.09, name,
                transform=ax.get_xaxis_transform(),
                ha='center', va='center', fontsize=6,
                fontweight='bold', color='white', clip_on=False)

# ─── Load RMSF data ───────────────────────────────────────────────────────────
systems = {
    '8RJJ-native': [
        f'{BASE}/salieri-data/8RJJ-native_05_minimized_replica_0{i}_rmsf.csv'
        for i in [1, 2, 3]
    ],
    '8RJJ-db411-424': [
        f'{BASE}/albicastro-data/8RJJ-m_I411C_S424C_db411-424/8RJJ-m_I411C_S424C_db411-424_05_minimized_replica_0{i}_rmsf.csv'
        for i in [1, 2, 3]
    ],
    '8RJJ-db424-429': [
        f'{BASE}/beethoven-data/8RJJ-m_I411C_S424C_db424-429/8RJJ-m_I411C_S424C_db424-429_05_minimized_replica_0{i}_rmsf.csv'
        for i in [1, 2, 3]
    ],
}

colors = {
    '8RJJ-native':   '#E74C3C',
    '8RJJ-db411-424':'#2ECC71',
    '8RJJ-db424-429':'#3498DB',
}

data = {}
for name, files in systems.items():
    reps = []
    for f in files:
        df = pd.read_csv(f)
        reps.append(df['rmsf_A'].values)
    arr = np.array(reps)
    residues = pd.read_csv(files[0])['residue'].values
    data[name] = {
        'residues': residues + SEQ_OFFSET,  # convert to E2 numbering
        'mean': arr.mean(axis=0),
        'std':  arr.std(axis=0),
    }
    print(f"{name}: {arr.shape[1]} residues, mean RMSF = {arr.mean():.2f} A")

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
fig.subplots_adjust(bottom=0.22)

for name, d in data.items():
    x    = d['residues']
    mean = d['mean']
    std  = d['std']
    col  = colors[name]
    ax.plot(x, mean, color=col, lw=1.5, label=name, zorder=3)
    ax.fill_between(x, mean - std, mean + std, color=col, alpha=0.15, zorder=2)

ax.set_xlabel('E2 residue number', labelpad=30)
ax.set_ylabel('Ca RMSF (A)')
ax.set_title('RMSF (Ca) per residue: mean ± SD across replicates\nImplicit solvent (GBneck2), ~1M frames per system')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

x_min = min(d['residues'].min() for d in data.values())
x_max = max(d['residues'].max() for d in data.values())
ax.set_xlim(x_min - 1, x_max + 1)

add_domain_bar(ax, x_min, x_max)

plt.savefig(os.path.join(OUTDIR, 'comparative_rmsf.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTDIR}/comparative_rmsf.png")
