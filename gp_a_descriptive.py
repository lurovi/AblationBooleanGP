"""Block A - descriptive layer: marginal means per level x n (with matched-pair significance), plateau shares."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gp_common import (FACTORS, NS, FACTOR_LABEL, PCT, Grid, level_labels, paired_pairs, dominance_marks,
                       to_tensor, heatmap, save_fig, save_table, print_latex, present_levels)

# column layout of the figure: factors stacked per column, heights proportional to #levels
LAYOUT = [['PopGenRatio', 'FunctionSet', 'TournamentSize'],
          ['Mutation', 'TreeInit', 'CrossMutRate', 'Crossover', 'Elites']]


def marginal_data(grid: Grid, target: str = 'score_clip'):
    """Returns {factor: (levels, mean_matrix[L, 10], marks[L][10])} for the chosen target array, with L being the number of levels for each factor. The mean matrix is the marginal mean of the target over all other factors, and marks are significance markers for each level pair (Holm-adjusted)."""
    V = getattr(grid, target)
    Y, levels = to_tensor(grid.cfg, V, FACTORS)
    out = {}
    for a, f in enumerate(FACTORS):
        other = tuple(k for k in range(len(FACTORS)) if k != a)
        means = Y.mean(axis=other)                               # (L, T)
        marks = [[''] * len(NS) for _ in levels[a]]
        for t in range(len(NS)):
            df = paired_pairs(Y[..., t], a)
            for l, m in enumerate(dominance_marks(df, len(levels[a]))):
                marks[l][t] = m
        out[f] = (levels[a], means, marks)
    return out


def marginal_heatmaps_callback(data, title_target: str = 'mean score'):
    fig = plt.figure(figsize=(7.4, 9.6), layout='constrained')
    outer = fig.add_gridspec(2, 2, height_ratios=[1, 0.025])
    axes = []
    for c, col in enumerate(LAYOUT):
        sub = outer[0, c].subgridspec(len(col), 1, height_ratios=[len(data[f][0]) + 1.2 for f in col])
        for r, f in enumerate(col):
            ax = fig.add_subplot(sub[r, 0])
            lv, means, marks = data[f]
            annot = [[f'{means[i, t]:.2f}{marks[i][t]}' for t in range(len(NS))] for i in range(len(lv))]
            im = heatmap(ax, means, [str(n) for n in NS], level_labels(f, lv), cmap='RdBu', vmin=-1, vmax=1,
                         annot=annot, fontsize=8.5)
            ax.set_title(FACTOR_LABEL[f], fontsize=10, loc='left', pad=2)
            ax.tick_params(axis='y', labelsize=8)
            if r == len(col) - 1:
                ax.set_xlabel(r'$n$', fontsize=10)
                ax.tick_params(axis='x', labelsize=8)
            else:
                ax.tick_params(axis='x', labelbottom=False)
            axes.append(ax)
    cb = fig.colorbar(im, cax=fig.add_subplot(outer[1, :]), orientation='horizontal')
    cb.set_label(f'{title_target} (0 = random balanced, 1 = best known)', fontsize=10)
    cb.ax.tick_params(labelsize=8)
    return fig


def marginal_heatmaps(grid: Grid, target: str = 'score_clip') -> dict:
    """Saves a figure with marginal heatmaps of the chosen target array (default: score_clip) and a table with the marginal means per level x n."""
    data = marginal_data(grid, target)
    save_fig(marginal_heatmaps_callback(data), 'fig_marginal_heatmaps')
    rows = [(f, l, *data[f][1][i]) for f in FACTORS for i, l in enumerate(data[f][0])] # flattened rows for the table
    save_table(pd.DataFrame(rows, columns=['factor', 'level'] + [f'N{n}' for n in NS]).set_index(['factor', 'level']), 'marginal_means_clip')
    return data


def plateau_shares(grid: Grid, ns=(8, 11, 14)) -> pd.DataFrame:
    """Share of configurations whose MEDIAN run ended on a balanced linear function (failure plateau), per level."""
    rows = {}
    cols = [NS.index(n) for n in ns]
    for f in FACTORS:
        for l in present_levels(grid.cfg, f):
            m = (grid.cfg[f] == l).to_numpy()
            rows[(FACTOR_LABEL[f], level_labels(f, [l])[0])] = [100 * grid.plateau[m][:, c].mean() for c in cols] + [100 * grid.plateau[m].mean()]
    df = pd.DataFrame(rows, index=[f'$n={n}$' for n in ns] + ['all $n$']).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=['factor', 'level'])
    return df


def run(grid: Grid) -> None:
    marginal_heatmaps(grid)
    pl = plateau_shares(grid)
    save_table(pl, 'plateau_shares')
    print_latex(pl, f'Share ({PCT}) of configurations whose median run ends on a balanced linear function '
                    '(failure plateau), by factor level.', 'tab:plateau', fmt='%.1f')
    tot = grid.plateau.mean(axis=0) * 100
    print('plateau share over the grid per n:', dict(zip(NS, np.round(tot, 2))))
    print('share of clipped (score < -1) cells per n:', dict(zip(NS, np.round((grid.score < -1).mean(0) * 100, 2))))