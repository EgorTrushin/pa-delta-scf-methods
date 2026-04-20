#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.labelsize': 9.5,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'figure.dpi': 200,
})

BASE = '/home/trushin/Projects/DeltaSCF/csv'

scatter_HSE12 = pd.read_csv(f'{BASE}/QUEST1_lowest_HSE12.csv', index_col=0)
tddft_q1 = pd.read_csv(f'{BASE}/QUEST1-TDDFT.csv')

C = {
    'UKS':      '#D95F5F',
    'pa-KS':    '#3A9E6E',
    'TDDFT':    '#E09030',
}

fig, axes = plt.subplots(1, 3, figsize=(5.25, 2.5), sharey=True)

calc_UKS_S = scatter_HSE12['UKS S'].values
calc_pa_S  = scatter_HSE12['pa-STA-KS S'].values
tbe_S      = scatter_HSE12['TBE S'].values
calc_UKS_T = scatter_HSE12['UKS T'].values
calc_pa_T  = scatter_HSE12['pa-STA-KS T'].values
tbe_T      = scatter_HSE12['TBE T'].values

# Match TDDFT to the same 16 molecules (first occurrence per molecule)
mol_order = scatter_HSE12.index.tolist()
tddft_matched_S, tddft_matched_T = [], []
tbe_tddft_S, tbe_tddft_T = [], []
tddft_q1_mol = tddft_q1['Molecule'].tolist()
seen = []
for mol in mol_order:
    for i, m in enumerate(tddft_q1_mol):
        if m.lower() == mol.lower() and i not in seen:
            tddft_matched_S.append(tddft_q1.iloc[i]['HSE12 S'])
            tddft_matched_T.append(tddft_q1.iloc[i]['HSE12 T'])
            tbe_tddft_S.append(tddft_q1.iloc[i]['TBE S'])
            tbe_tddft_T.append(tddft_q1.iloc[i]['TBE T'])
            seen.append(i)
            break

tddft_S   = np.array(tddft_matched_S)
tddft_T   = np.array(tddft_matched_T)
tbe_td_S  = np.array(tbe_tddft_S)
tbe_td_T  = np.array(tbe_tddft_T)

# Classify excitation types
exc_col = scatter_HSE12['excitation'].values
exc_type = ['Rydberg' if 'R' in str(e) else 'Valence' for e in exc_col]
exc_colors = ['#4A7EC4' if t == 'Rydberg' else '#E05C5C' for t in exc_type]

emin, emax = 2.0, 11.0

panels = [
    ('UKS',
     np.concatenate([calc_UKS_S, calc_UKS_T]),
     np.concatenate([tbe_S, tbe_T]),
     C['UKS']),
    ('pa-STA-KS\n(this work)',
     np.concatenate([calc_pa_S, calc_pa_T]),
     np.concatenate([tbe_S, tbe_T]),
     C['pa-KS']),
    ('TDDFT',
     np.concatenate([tddft_S, tddft_T]),
     np.concatenate([tbe_td_S, tbe_td_T]),
     C['TDDFT']),
]

for ax, (label, calc, tbe_ref, color) in zip(axes, panels):
    mae_val = np.mean(np.abs(calc - tbe_ref))
    dot_colors = (exc_colors * 2)[:len(calc)]

    ax.scatter(tbe_ref, calc, c=dot_colors, s=20, alpha=0.85,
               edgecolors='white', linewidths=0.5, zorder=4)
    ax.plot([emin, emax], [emin, emax], 'k-', lw=1.2, zorder=3, alpha=0.5)
    ax.set_xlim(emin, emax)
    ax.set_ylim(emin, emax)
    ax.set_title(label, fontsize=9, fontweight='bold', color=color)
    ax.text(0.97, 0.04, f'MAE = {mae_val:.2f} eV', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8.0,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.18, edgecolor=color))
    ax.set_xlabel('Reference (eV)', fontsize=9)
    ax.yaxis.grid(True, alpha=0.25, linestyle='--')
    ax.xaxis.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Calculated (eV)', fontsize=9)

legend_els = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#E05C5C',
                      markersize=5, label='Valence'),
              Line2D([0], [0], marker='o', color='w', markerfacecolor='#4A7EC4',
                      markersize=5, label='Rydberg')]
axes[2].legend(handles=legend_els, fontsize=7.5, loc='upper left',
               framealpha=0.8, edgecolor='lightgray')

fig.suptitle('Excitation energies: calculated vs. reference (HSE12, QUEST1)',
             fontsize=10, y=0.97)
fig.tight_layout(pad=0.5)
plt.savefig('graphical_abstract.pdf', format='pdf', bbox_inches='tight')
