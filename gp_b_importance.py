"""
Block B - exact fANOVA (functional ANOVA on the complete categorical grid) and exact Shapley effects.

The grid is a complete product of 8 categorical factors, so the decomposition is computed exactly from
marginal means (no random-forest surrogate). Uniform distribution over the levels of every factor.
Targets: 'clip' (main; anchored score clipped at -1), 'rank' (within-n percentile rank), 'raw' (unclipped score).

For one problem size n there are 25,200 numbers, one clipped score per configuration. They are arranged in an 8-dimensional box with one axis per knob (3 x 7 x 3 x 2 x 4 x 5 x 5 x 2).
The question is how much of the spread (variance) of these numbers is due to each knob. A knob is important when changing it moves the score a lot, on average over all the other settings.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gp_common import (FACTORS, NS, FACTOR_LABEL, PCT, Grid, to_tensor, heatmap, save_fig, save_table, print_latex)

D = len(FACTORS)
TARGETS = {'clip': 'score_clip', 'rank': 'rank', 'raw': 'score', 'logstd': 'log_std'}


def anova_components(Y: np.ndarray) -> tuple[dict, np.ndarray]:
    """
    Y: (d_1..d_D, T). Returns {subset(tuple of axes): variance component (T,)} and total variance (T,). 
    """
    T = Y.shape[-1]
    gm = Y.mean(axis=tuple(range(D)))
    closed = {(): np.zeros(T)}
    for r in range(1, D + 1):
        for u in itertools.combinations(range(D), r):
            comp = tuple(a for a in range(D) if a not in u)
            m = Y.mean(axis=comp) if comp else Y
            closed[u] = (m ** 2).mean(axis=tuple(range(r))) - gm ** 2          # Var(E[Y | x_u])
    comps = {}
    for u in closed:
        s = np.zeros(T)
        for r in range(len(u) + 1):
            for v in itertools.combinations(u, r):
                s += (-1) ** (len(u) - r) * closed[v]                           # Moebius inversion
        comps[u] = s
    V = closed[tuple(range(D))]
    assert np.allclose(sum(comps.values()), V, rtol=1e-8, atol=1e-12), 'components do not sum to total variance'
    return comps, V


def importance_tables(grid: Grid, target: str = 'clip') -> dict:
    """
    Compute importance tables for a given grid and target.
    Returns a dictionary containing:
    - terms: DataFrame of variance components for each factor combination (terms) and their order
    - shapley: DataFrame of Shapley effects for each factor
    - main: DataFrame of main effects for each factor
    - total: DataFrame of total effects for each factor
    - by_order: DataFrame of variance explained by order (1-way, 2-way, 3-way, >=4-way)
    - noise: Series of estimated noise share from the std file
    - noise_hi: Series of estimated noise share from the 8-way term
    - noise_by_order: DataFrame of estimated noise share by order
    - dof_order: Series of degrees of freedom by order
    - V: Series of total variance for each problem size (n)

    - to_tensor builds the 8-dimensional box and checks that it is complete.
    - anova_components splits the total variance V into pieces, one for each subset of knobs. There are 255 non-empty subsets. In two stages:
    - closed[u] is the variance of the averages you get when you keep only the knobs in u and average over all the others. It is the variance explained by u and everything inside it.
      The pure piece of u is closed[u] minus the pieces of all its smaller subsets (inclusion-exclusion, the “Möbius inversion” in the comment).
      For a pair this means: “variance of the pair” minus “variance of knob A alone” minus “variance of knob B alone”.
      What is left is what only appears when A and B act together. The assert checks that all pieces add up to V.
    - terms is each pure piece divided by V, so every term is a percentage of the total.
    - main, total, shapley each reduce the terms to one number per knob (next section).
    - by_order adds up all terms with the same number of knobs: 1-way, 2-way, 3-way, and 4-way or higher.

    A toy example with two knobs, A and B, and scores 0, 0, 0, 4 for A1B1, A1B2, A2B1, A2B2. The total variance is 3. Knob A alone explains 1, B alone explains 1, and the remaining 1 appears only when both are at level 2 (the interaction).

    Measure	Definition	Toy result for A
        Main	the knob's own piece only	1/3 = 33%
        Total	its own piece plus every interaction it is part of	(1 + 1)/3 = 67%
        Shapley	own piece plus an equal share of each interaction it takes part in	(1 + 1/2)/3 = 50%
    Main ≤ Shapley ≤ total, always. Shapley effects add up to exactly 100% (A and B get 50% each). Total effects add up to more than 100%, because each interaction is counted for every knob in it.
    Main effects ignore interactions altogether. That is why a knob involved only in interactions shows a small main effect but a large Shapley effect. total is computed and saved but not plotted.
    """
    Y, levels = to_tensor(grid.cfg, getattr(grid, TARGETS[target]), FACTORS)
    comps, V = anova_components(Y)
    names = lambda u: ' x '.join(FACTOR_LABEL[FACTORS[a]] for a in u)
    terms = pd.DataFrame({names(u): c / V for u, c in comps.items() if u}, index=NS).T
    terms['order'] = [len(u) for u in comps if u]
    shap = pd.DataFrame({FACTORS[i]: sum(c / len(u) for u, c in comps.items() if i in u) / V for i in range(D)},
                        index=NS).T
    main = pd.DataFrame({FACTORS[i]: comps[(i,)] / V for i in range(D)}, index=NS).T
    total = pd.DataFrame({FACTORS[i]: sum(c for u, c in comps.items() if i in u) / V for i in range(D)}, index=NS).T
    by_order = terms.groupby('order')[NS].sum()
    by_order.loc['>=4'] = by_order.loc[by_order.index >= 4].sum()
    by_order = by_order.loc[[1, 2, 3, '>=4']]
    assert np.allclose(shap.sum(), 1.0), 'Shapley effects must sum to 1'
    # ---- noise ---------------------------------------------------------------------------------
    # Observed value of a cell = true value + sampling noise of the median of 20 runs. Noise is "white" over the
    # cells, so it spreads over the ANOVA terms in proportion to their degrees of freedom:
    #   dof(u) = prod_{i in u} (L_i - 1),   E[variance component of u due to noise] = s2 * dof(u) / ncell.
    dof = {u: float(np.prod([len(levels[a]) - 1 for a in u])) for u in comps if u}
    ncell = Y.size // Y.shape[-1]                                           # number of grid cells (25,200)
    dof_order = pd.Series({o: sum(d for u, d in dof.items() if len(u) == o) for o in range(1, D + 1)})
    # (A) noise from the std file: variance of a median of 20 runs ~ (pi/2) sigma^2 / 20 (normal approximation).
    #     Probably an OVERestimate: sigma includes rare catastrophic runs that barely move a median.
    sd = np.minimum(grid.std_score, 1.0) if target == 'clip' else (grid.std_score if target == 'raw' else None)
    noise_share = pd.Series(np.nan if sd is None else (np.pi / 2) * (sd ** 2).mean(axis=0) / 20.0 / V, index=NS)
    # (B) noise from the data itself: the highest-order term (all 8 factors) is assumed to contain only noise
    #     (classical ANOVA without replication). If a true 8-way interaction exists this is an upper bound.
    full = tuple(range(D))
    s2 = comps[full] * ncell / dof[full]                                    # per-cell noise variance
    noise_hi = pd.Series(s2 * (ncell - 1) / ncell / V, index=NS)
    noise_by_order = pd.DataFrame({o: s2 * dof_order[o] / ncell / V for o in range(1, D + 1)}, index=NS).T
    noise_by_order.loc['>=4'] = noise_by_order.loc[noise_by_order.index >= 4].sum()
    noise_by_order = noise_by_order.loc[[1, 2, 3, '>=4']]
    return dict(terms=terms, shapley=shap, main=main, total=total, by_order=by_order, noise=noise_share, noise_hi=noise_hi,
                noise_by_order=noise_by_order, dof_order=dof_order, V=pd.Series(V, index=NS))


# ------------------------------------------------------------------ plots
def fig_orders_callback(res):
    """
    Plot the order of importance of the factors in the model.
    left: Each bar is one n. The colours show the share of variance coming from main effects, pairs, triples, and 4-way or higher. Every bar sums to 100%.
    right: Compares the share in the 4-way and higher bin with two noise estimates. It checks whether those high-order terms can be told apart from noise.
    """
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), layout='constrained', gridspec_kw={'width_ratios': [1.35, 1]})
    bo = res['by_order'] * 100
    bottom = np.zeros(len(NS))
    cols = ['#0072B2', '#56B4E9', '#E69F00', '#999999']
    labels = ['main effects', '2-way', '3-way', r'$\geq 4$-way (+ noise)']
    for (idx, row), c, lab in zip(bo.iterrows(), cols, labels):
        ax[0].bar(range(len(NS)), row.values, bottom=bottom, color=c, label=lab, edgecolor='black', linewidth=0.4)
        bottom += row.values
    ax[0].set_xticks(range(len(NS)), [str(n) for n in NS])
    ax[0].set_xlabel(r'$n$', fontsize=10)
    ax[0].set_ylabel(f'Variance explained ({PCT})', fontsize=10)
    ax[0].set_ylim(0, 100)
    ax[0].legend(fontsize=8, ncol=2, loc='lower center', frameon=True, framealpha=0.9)
    ax[0].grid(True, axis='y', color='gray', linestyle='--', linewidth=0.5)
    ax[0].set_axisbelow(True)
    ax[1].plot(NS, bo.loc['>=4'].values, 'o-', color='#999999', label=r'$\geq 4$-way share')
    ax[1].plot(NS, res['noise_hi'].values * 100, 's--', color='black', label='noise (from 8-way term)')
    ax[1].plot(NS, res['noise'].values * 100, '^:', color='#D55E00', label='noise (from std file)')
    ax[1].set_xlabel(r'$n$', fontsize=10)
    ax[1].set_ylabel(f'Share of variance ({PCT})', fontsize=10)
    ax[1].legend(fontsize=8)
    ax[1].grid(True, color='gray', linestyle='--', linewidth=0.5)
    return fig


def fig_shapley_callback(res):
    """
    Plot the Shapley effects of the factors in the model.
    Left: Shapley effects (columns sum to 100%). Right: main effects only (columns sum to the main-effect total, 68-86%). The gap between the two for a knob is the part of the interactions credited to it.
    """
    order = res['shapley'].mean(axis=1).sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), layout='constrained', sharey=True)
    lab = [FACTOR_LABEL[f] for f in order]
    vmax = res['shapley'].to_numpy().max() * 100
    for a, key, title in zip(ax, ['shapley', 'main'], ['Shapley effect', 'Main effect only']):
        M = res[key].loc[order].to_numpy() * 100
        im = heatmap(a, M, [str(n) for n in NS], lab, cmap='cividis', vmin=0, vmax=vmax, fmt='{:.0f}', fontsize=10)
        a.set_title(f'{title} ({PCT})', fontsize=10)
        a.set_xlabel(r'$n$', fontsize=10)
    ax[1].tick_params(axis='y', labelleft=False)
    cb = fig.colorbar(im, ax=ax, location='right', shrink=0.9, pad=0.01)
    cb.ax.tick_params(labelsize=8)
    return fig


def fig_interactions_callback(res, top_pairs=12, top_triples=8):
    """
    The 12 largest pure pair terms and the 8 largest pure triple terms, each as a percentage of the total variance. A value of 3% for a pair means that pair accounts for 3% of the variance beyond what the two knobs explain separately.
    """
    t = res['terms']
    sel = {}
    for o, k in ((2, top_pairs), (3, top_triples)):
        s = t[t.order == o]
        sel[o] = (s[NS].loc[s[NS].mean(axis=1).sort_values(ascending=False).index[:k]] * 100)
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 5.4), layout='constrained',
                           gridspec_kw={'height_ratios': [top_pairs, top_triples]})
    vmax = max(sel[2].to_numpy().max(), sel[3].to_numpy().max())
    for a, o in zip(ax, (2, 3)):
        im = heatmap(a, sel[o].to_numpy(), [str(n) for n in NS], [s.replace(' x ', r'$\times$') for s in sel[o].index],
                     cmap='cividis', vmin=0, vmax=vmax, fmt='{:.1f}', fontsize=10)
        a.tick_params(axis='y', labelsize=8)
        a.set_title(f'Top {len(sel[o])} {o}-way terms ({PCT} of variance)', fontsize=10, loc='left')
    ax[1].set_xlabel(r'$n$', fontsize=10)
    ax[0].tick_params(axis='x', labelbottom=False)
    cb = fig.colorbar(im, ax=ax, location='right', shrink=0.9, pad=0.01)
    cb.ax.tick_params(labelsize=8)
    return fig


# ------------------------------------------------------------------ driver
def run(grid: Grid) -> dict:
    out = {k: importance_tables(grid, k) for k in TARGETS}
    res = out['clip']
    for k, r in out.items():
        save_table(r['shapley'], f'shapley_{k}')
        save_table(r['main'], f'fanova_main_{k}')
        save_table(r['terms'], f'fanova_terms_{k}')
    save_fig(fig_orders_callback(res), 'fig_fanova_orders')
    save_fig(fig_shapley_callback(res), 'fig_shapley_heatmap')
    save_fig(fig_interactions_callback(res), 'fig_fanova_interactions')

    # LaTeX: Shapley effects (%), best per column in bold
    S = (res['shapley'] * 100)
    S = S.loc[S.mean(axis=1).sort_values(ascending=False).index]
    tab = S.copy().astype(object)
    for n in NS:
        top = S[n].idxmax()
        for f in S.index:
            tab.loc[f, n] = (f'\\textbf{{{S.loc[f, n]:.1f}}}' if f == top else f'{S.loc[f, n]:.1f}')
    tab.index = [FACTOR_LABEL[f] for f in tab.index]
    tab.columns = [f'$n={n}$' for n in NS]
    print_latex(tab, f'Exact Shapley effects ({PCT} of the variance of the clipped score) per problem size.',
                'tab:shapley', fmt='%s')
    print('\nvariance explained by order (clip target, %):\n', (res['by_order'] * 100).round(1).to_string())
    print('\nestimated noise share (% of variance), from the std file vs from the 8-way term:\n',
          pd.DataFrame({'std file': res['noise'], '8-way term': res['noise_hi']}).mul(100).round(2).T.to_string())
    print('\ndegrees of freedom by order:', res['dof_order'].astype(int).to_dict())
    print('expected noise inside each order (% of variance):\n', (res['noise_by_order'] * 100).round(2).to_string())
    print('observed minus expected noise, i.e. estimated signal (% of variance):\n',
          ((res['by_order'] - res['noise_by_order']) * 100).round(2).to_string())
    # robustness of the Shapley ranking across targets
    rob = pd.DataFrame({k: out[k]['shapley'].mean(axis=1) for k in TARGETS})
    print('\nmean Shapley share by target:\n', (rob * 100).round(1).to_string())
    print('Spearman between targets (mean Shapley):\n', rob.corr(method='spearman').round(2).to_string())
    return out