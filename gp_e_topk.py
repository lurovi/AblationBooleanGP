"""
Block E - how much of the attainable improvement is recovered by tuning only k hyperparameters?

value(S) = E_{x_rest ~ uniform} [ max_{x_S} score(x) ]  : best score reachable by tuning the set S while the
remaining factors sit at an unknown (uniformly random) level.  gap_closed(S) = (value(S)-value(empty)) / (value(all)-value(empty)).
value(empty) = mean over the grid (random configuration), value(all) = best configuration.
Computed exactly on the 25,200-cell grid for all 2^8 subsets.


Suppose you can only tune k of the 8 knobs and the rest sit at an unknown setting. How much of the possible improvement do you still get? The code computes, for each chosen set S of knobs:

For every combination of the other knobs, take the best score reachable by choosing S's levels freely.
Average that best score over all combinations of the others (value(S)).
Convert it to “gap closed”: (value(S) - mean of the grid) / (best configuration - mean of the grid).

Zero means no better than a random configuration and 100% means the best one. This is computed exactly for all 256 subsets at each n, using the clipped score.

all_subset_values takes the maximum over the axes in S and the mean over the others, for each subset.
gap_closed_table normalises with the grid mean (S empty) and the best configuration (all knobs).
strategies asks which subset to pick:
    the top-k by Shapley effect at each n,
    the top-k by the Shapley effect averaged over n,
    the top-k by main effect,
    the oracle, the best subset of that size at each n, which is the upper bound for any rule.
best_subsets lists the oracle subsets (printed for n = 8, 11, 14).
The figure. The left panel shows one curve per n for the Shapley rule. The right panel shows the mean over n for the four strategies. The points at k = 0 (0%) and k = 8 (100%) hold by construction.

"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gp_common import FACTORS, NS, FACTOR_LABEL, PCT, Grid, to_tensor, n_colors, save_fig, save_table, print_latex

D = len(FACTORS)


def all_subset_values(Y: np.ndarray) -> dict:
    """{subset (tuple of factor indices): value(S) (array of length 10)}."""
    out = {}
    for r in range(D + 1):
        for S in itertools.combinations(range(D), r):
            m = Y.max(axis=S) if S else Y
            out[S] = m.mean(axis=tuple(range(m.ndim - 1)))                 # (T,)
    return out


def gap_closed_table(grid: Grid) -> dict:
    """{subset (tuple of factor indices): gap closed per n (array of length 10)}."""
    Y, _ = to_tensor(grid.cfg, grid.score_clip, FACTORS)
    val = all_subset_values(Y)
    v0, v1 = val[()], val[tuple(range(D))]
    return {S: (v - v0) / (v1 - v0) for S, v in val.items()}


def strategies(gc: dict, shapley: pd.DataFrame, main: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """gap closed vs k for: n-specific Shapley ranking, global Shapley ranking, main-effect ranking, oracle subset."""
    idx = {f: i for i, f in enumerate(FACTORS)}
    glob = shapley.mean(axis=1).sort_values(ascending=False).index.tolist()
    res = {k: pd.DataFrame(index=range(D + 1), columns=NS, dtype=float) for k in
           ['shapley_n', 'shapley_global', 'main_n', 'oracle']}
    for t, n in enumerate(NS):
        rank_n = shapley[n].sort_values(ascending=False).index.tolist()
        rank_m = main[n].sort_values(ascending=False).index.tolist()
        for k in range(D + 1):
            get = lambda order: gc[tuple(sorted(idx[f] for f in order[:k]))][t]
            res['shapley_n'].loc[k, n] = get(rank_n)
            res['shapley_global'].loc[k, n] = get(glob)
            res['main_n'].loc[k, n] = get(rank_m)
            res['oracle'].loc[k, n] = max(v[t] for S, v in gc.items() if len(S) == k)
    return res


def best_subsets(gc: dict) -> pd.DataFrame:
    """List the best subset of each size k for each n, with the gap closed and the factor names."""
    rows = []
    for k in range(1, D):
        for t, n in enumerate(NS):
            S = max((S for S in gc if len(S) == k), key=lambda S: gc[S][t])
            rows.append(dict(k=k, n=n, gap_closed=gc[S][t], subset=', '.join(FACTOR_LABEL[FACTORS[a]] for a in S)))
    return pd.DataFrame(rows)


def fig_topk_callback(res):
    """Create a figure showing the top-k strategies."""
    col = n_colors()
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), layout='constrained', sharey=True)
    for n in NS:
        ax[0].plot(range(D + 1), res['shapley_n'][n] * 100, 'o-', ms=3, lw=1.2, color=col[n], label=f'$n={n}$')
    ax[0].set_title('Tune top-$k$ by Shapley (per $n$)', fontsize=10)
    sty = {'shapley_n': ('-', 'o', '#0072B2', r'Shapley ($n$-specific)'),
           'shapley_global': ('--', 's', '#D55E00', 'Shapley (global)'),
           'main_n': (':', '^', '#009E73', 'main effect'),
           'oracle': ('-', 'D', 'black', 'oracle subset')}
    for k, (ls, mk, c, lab) in sty.items():
        ax[1].plot(range(D + 1), res[k].mean(axis=1) * 100, ls=ls, marker=mk, ms=3.5, color=c, label=lab)
    ax[1].set_title(r'Mean over $n$', fontsize=10)
    for a in ax:
        a.set_xlabel(r'number of tuned hyperparameters $k$', fontsize=10)
        a.grid(True, color='gray', linestyle='--', linewidth=0.5)
        a.set_xticks(range(D + 1))
    ax[0].set_ylabel(f'Attainable gap closed ({PCT})', fontsize=10)
    ax[0].legend(fontsize=8, ncol=2, loc='lower right')
    ax[1].legend(fontsize=8, loc='lower right')
    return fig


def run(grid: Grid, importance: dict) -> dict:
    gc = gap_closed_table(grid)
    r = importance['clip']
    res = strategies(gc, r['shapley'], r['main'])
    save_fig(fig_topk_callback(res), 'fig_topk_regret')
    for k, df in res.items():
        save_table(df, f'topk_{k}')
    mean = pd.DataFrame({k: v.mean(axis=1) * 100 for k, v in res.items()}).loc[1:D - 1]
    mean.index.name = 'k'
    mean.columns = ['Shapley ($n$-spec.)', 'Shapley (global)', 'Main effect', 'Oracle']
    print_latex(mean, f'Attainable gap closed ({PCT}), averaged over $n$, when tuning only $k$ hyperparameters '
                      '(others at random levels).', 'tab:topk', fmt='%.1f')

    d = (mean.iloc[:, :3].max(axis=1) - mean.iloc[:, :3].min(axis=1)).max()
    print(f'largest difference between the three ranking strategies at any k: {d:.3f} percentage points')
    if d < 0.1:
        print('NOTE: the three rankings are practically equivalent: report one of them next to the oracle.')

    glob = r['shapley'].mean(axis=1).sort_values(ascending=False)
    print('global Shapley ranking:', [FACTOR_LABEL[f] for f in glob.index])

    # per problem size: how many knobs are needed? (the gap is measured against the best median of the grid)
    by_n = res['shapley_n'].loc[1:D - 1] * 100
    by_n.index.name = 'k'
    by_n.columns = [f'$n={n}$' for n in NS]
    print_latex(by_n, f'Attainable gap closed ({PCT}) when tuning the top-$k$ hyperparameters (Shapley ranking), '
                      'per problem size.', 'tab:topk_by_n', fmt='%.1f')
    need = {n: next((k for k in range(D + 1) if res['shapley_n'].loc[k, n] >= 0.99), D) for n in NS}
    print('smallest k (Shapley ranking) closing >= 99% of the gap, per n:', need)
    bs = best_subsets(gc)
    save_table(bs.set_index(['k', 'n']), 'topk_best_subsets')
    print('\nbest subset per k for n = 8, 11, 14:\n', bs[bs.n.isin([8, 11, 14])].round(3).to_string(index=False))
    return res
