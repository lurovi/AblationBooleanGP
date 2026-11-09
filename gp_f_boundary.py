"""
Block F - boundary regimes on the deduplicated 5-rate grid (37,800 cfgs; Crossover = NotUsed under OnlyMutation).
Reported descriptively: score vs n per rate, and operator effects per regime (matched pairs).
Reminder for the text: with Cx prob. 1.0 and an odd number of non-elite offspring the unpaired last offspring is mutated every generation, so Mutation is still (weakly) active under OnlyCrossover; under OnlyMutation crossover is never called.

The main analysis uses only the three mixed rates. This block looks at the five-rate grid, which also contains the two boundary cases, where only one operator is used (the probabilities are of choosing crossover versus mutation at each reproduction step).

Left panel: the mean clipped score of each rate as a function of n.
Right panel, operator effects: for each regime, how much better or worse each operator is than its reference (Uniform mutation, one-point crossover).
How an effect is computed: compare configurations that are identical in every other knob (matched pairs), average the score difference over n ≥ 8, and report the mean over pairs. A star requires a Holm-adjusted p below 0.05 and that the operator is better (or worse) in at least 56% of the pairs, with ties counted half.
Why n ≥ 8: below that the scores are coarse and tied at the ceiling.
The grid: under OnlyMutation the crossover operator is never called, so that regime uses the deduplicated rows (4,200 instead of 8,400 configurations). The means are not distorted, because each crossover level was equally represented.

Each Cx/Mut setting is compared with 0.5/0.5 on matched configurations (under OnlyMutation the reference is averaged over its two crossover operators). The Holm correction uses a family of four comparisons. The output is a new table, tab:boundary_vs_mixed, with pairs, mean difference, the share of pairs where the setting wins, and a star.

With a crossover probability of 1.0 and an odd number of non-elite offspring, the last offspring is mutated in every generation. So mutation is weakly active under OnlyCrossover. That is why the operator effects there are about half those in the mixed regimes.
The matched pairs share the same 20 seeds across configurations, so the p-values are screening devices. Rely on the mean differences and the share of pairs.
The scores are clipped at -1, so some of the differences are about avoiding failure and not about finer quality differences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gp_common import (FACTORS, NS, ALL_RATES, OKABE, LEVEL_LABEL, PCT, AST, Grid, to_tensor, holm_adjust,
                       heatmap, save_fig, save_table, print_latex)
from scipy import stats

REGIMES = ['OnlyMutation', 'LowCrossoverHighMutation', 'EqualCrossoverMutation', 'HighCrossoverLowMutation',
           'OnlyCrossover']                                   # increasing crossover probability: 0.0, 0.3, 0.5, 0.7, 1.0
N_MIN = 8
ROWS = [('Mutation', 'MutInsert'), ('Mutation', 'MutNodeReplacement'), ('Mutation', 'MutShrink'),
        ('Crossover', 'CxOnePointLeafBiased')]
REF = {'Mutation': 'MutUniform', 'Crossover': 'CxOnePoint'}

_NAMES = {'MutInsert': 'Insert', 'MutNodeReplacement': 'Node repl.', 'MutShrink': 'Shrink', 'CxOnePointLeafBiased': 'Leaf-biased'}
ROW_LABELS = [f'{f}: {_NAMES[l]} vs {"Uniform" if f == "Mutation" else "One-point"}' for f, l in ROWS]



def operator_effects(grid: Grid, n_min: int = N_MIN, alpha=0.05, a_min=0.56):
    """Per regime and operator: mean over matched pairs of (mean over n>=n_min of score difference vs reference)."""
    cols = [i for i, n in enumerate(NS) if n >= n_min]
    others = [f for f in FACTORS if f != 'CrossMutRate']
    eff = pd.DataFrame(np.nan, index=[f'{f}:{l}' for f, l in ROWS], columns=REGIMES)
    pv = eff.copy()
    Aeff = eff.copy()
    for reg in REGIMES:
        sub = grid.cfg['CrossMutRate'] == reg
        Y, lv = to_tensor(grid.cfg[sub].reset_index(drop=True), grid.score_clip[sub.to_numpy()][:, cols], others)
        for f, l in ROWS:
            a = others.index(f)
            if l not in lv[a] or REF[f] not in lv[a]:
                continue
            d = (np.take(Y, lv[a].index(l), axis=a) - np.take(Y, lv[a].index(REF[f]), axis=a)).mean(-1).ravel()
            nz = d[d != 0]
            key = f'{f}:{l}'
            eff.loc[key, reg] = d.mean()
            pv.loc[key, reg] = stats.wilcoxon(nz).pvalue if nz.size >= 10 else 1.0
            Aeff.loc[key, reg] = (d > 0).mean() + 0.5 * (d == 0).mean()
    ok = ~pv.isna().to_numpy().ravel()
    adj = np.full(pv.size, np.nan)
    adj[ok] = holm_adjust(pv.to_numpy().ravel()[ok])
    padj = pd.DataFrame(adj.reshape(pv.shape), index=pv.index, columns=pv.columns)
    sig = (padj < alpha) & ((Aeff - 0.5).abs() >= a_min - 0.5)
    return eff, Aeff, sig


def regime_vs_mixed(grid: Grid, ref='EqualCrossoverMutation', n_min: int = N_MIN, alpha=0.05, a_min=0.56) -> pd.DataFrame:
    """
    Tests the claim "mixing operators beats using one alone". Each Cx/Mut setting is compared with the reference (0.5/0.5)
    on matched configurations (identical in every other factor; under OnlyMutation, where Crossover is not used, the
    reference is averaged over its two crossover operators). Difference = mean over n >= n_min of the clipped score.
    Same decision rule as the operator table: Holm-adjusted p < alpha and |A - 0.5| >= a_min - 0.5.
    Returns a DataFrame with the number of pairs, mean difference, A, Holm-adjusted p-value, and significance.
    """
    cols = [i for i, n in enumerate(NS) if n >= n_min]
    others = [f for f in FACTORS if f != 'CrossMutRate']
    Y = {}
    for reg in REGIMES:
        sub = (grid.cfg['CrossMutRate'] == reg).to_numpy()
        Y[reg], _ = to_tensor(grid.cfg[sub].reset_index(drop=True), grid.score_clip[sub][:, cols], others)
    cx = others.index('Crossover')
    rows = []
    for reg in REGIMES:
        if reg == ref:
            continue
        Yref = Y[ref] if Y[reg].shape[cx] == Y[ref].shape[cx] else Y[ref].mean(axis=cx, keepdims=True)
        d = (Y[reg] - Yref).mean(-1).ravel()
        nz = d[d != 0]
        rows.append(dict(regime=reg, pairs=int(d.size), mean_diff=d.mean(), A=(d > 0).mean() + 0.5 * (d == 0).mean(),
                         p=stats.wilcoxon(nz).pvalue if nz.size >= 10 else 1.0))
    df = pd.DataFrame(rows).set_index('regime')
    df['p_holm'] = holm_adjust(df['p'].to_numpy())
    df['sig'] = (df['p_holm'] < alpha) & ((df['A'] - 0.5).abs() >= a_min - 0.5)
    return df


def fig_boundary_callback(grid: Grid, eff: pd.DataFrame, sig: pd.DataFrame):
    """Creates a figure with two subplots: one showing the mean scores for each regime, and another showing the operator effects."""
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), layout='constrained', gridspec_kw={'width_ratios': [1, 1.15]})
    for i, reg in enumerate(REGIMES):
        m = (grid.cfg['CrossMutRate'] == reg).to_numpy()
        ax[0].plot(NS, grid.score_clip[m].mean(0), 'o-', ms=3, color=OKABE[i],
                   label=LEVEL_LABEL['CrossMutRate'][reg])
    ax[0].set_xlabel(r'$n$', fontsize=10)
    ax[0].set_ylabel('mean score (clipped)', fontsize=10)
    ax[0].set_xticks(NS)
    ax[0].grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax[0].legend(title='Cx/Mut prob.', fontsize=8, title_fontsize=9)
    ylab = ROW_LABELS
    annot = [[('' if np.isnan(eff.iloc[i, j]) else f'{eff.iloc[i, j]:+.2f}' + (AST if sig.iloc[i, j] else ''))
              for j in range(len(REGIMES))] for i in range(len(ROWS))]
    lim = np.nanmax(np.abs(eff.to_numpy()))
    im = heatmap(ax[1], eff.to_numpy(), [LEVEL_LABEL['CrossMutRate'][r] for r in REGIMES], ylab, cmap='RdBu',
                 vmin=-lim, vmax=lim, annot=annot, fontsize=10)
    ax[1].set_xlabel('Cx/Mut prob.', fontsize=10)
    ax[1].set_title(rf'Operator effect (score diff., $n\geq{N_MIN}$)', fontsize=10)
    ax[1].tick_params(axis='y', labelsize=8)
    fig.colorbar(im, ax=ax[1], shrink=0.85, pad=0.01).ax.tick_params(labelsize=8)
    return fig


def run(grid: Grid) -> None:
    eff, A, sig = operator_effects(grid)
    save_fig(fig_boundary_callback(grid, eff, sig), 'fig_boundary_regimes')
    save_table(eff, 'boundary_operator_effects')
    save_table(A, 'boundary_operator_effects_A')
    tab = eff.copy().astype(object)
    for i in range(eff.shape[0]):
        for j in range(eff.shape[1]):
            v = eff.iloc[i, j]
            tab.iloc[i, j] = '--' if np.isnan(v) else f'{v:+.3f}' + (r'$^{*}$' if sig.iloc[i, j] else '')
    tab.columns = [LEVEL_LABEL['CrossMutRate'][r] for r in REGIMES]
    tab.index = ROW_LABELS
    print_latex(tab, rf'Mean paired score difference of each operator against the reference (Uniform mutation, one-point '
                     rf'crossover) for $n\geq{N_MIN}$; $^{{*}}$: Holm-adjusted $p<0.05$ and paired $A\geq0.56$ or $\leq0.44$.',
                'tab:boundary', fmt='%s')

    rv = regime_vs_mixed(grid)
    save_table(rv, 'boundary_regime_vs_mixed')
    tabr = pd.DataFrame({'pairs': rv['pairs'].astype(str),
                         'mean diff.': [f'{v:+.3f}' + (r'$^{*}$' if s_ else '') for v, s_ in zip(rv['mean_diff'], rv['sig'])],
                         '$A$': [f'{a:.2f}' for a in rv['A']]})
    tabr.index = [LEVEL_LABEL['CrossMutRate'][r] for r in rv.index]
    tabr.index.name = 'Cx/Mut prob.'
    print_latex(tabr, rf'Mean paired score difference of each Cx/Mut setting against 0.5/0.5 on matched configurations '
                      rf'($n\geq{N_MIN}$); $A$ = share of pairs in which the setting is better (ties count half); '
                      rf'$^{{*}}$: Holm-adjusted $p<0.05$ and $A\geq0.56$ or $\leq0.44$.',
                'tab:boundary_vs_mixed', fmt='%s')


    print('mean clipped score per rate and n:\n', pd.DataFrame(
        {r: grid.score_clip[(grid.cfg.CrossMutRate == r).to_numpy()].mean(0) for r in REGIMES}, index=NS).round(3).T.to_string())
