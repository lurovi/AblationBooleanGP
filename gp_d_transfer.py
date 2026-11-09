"""
Block D - meta-learning view: each problem size n is a task.
  (a) Spearman correlation of configuration rankings across n (raw medians, tie-aware)
  (b) Jaccard overlap of the top-5% configurations
  (c) normalized regret of transferring the best configuration of task s to task t (0 = optimal, 1 = random)
  (d) Kendall tau between Shapley importance profiles of different n
  + warm-start table: cheap fidelity (n = 5..9) -> expensive target (n = 12..14), best-of-k re-evaluation.

Each problem size n is treated as a separate task, and the question is whether a configuration that is good on one n is also good on another. If yes, you can tune on cheap small problems and reuse the result on expensive large ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

from gp_common import NS, PCT, CLIP_LOW, Grid, load_anchors, normalized_regret, heatmap, save_fig, save_table, print_latex

T = len(NS)


def spearman_matrix(grid: Grid) -> np.ndarray:
    """Spearman correlation of configuration rankings across n (raw medians, tie-aware)."""
    return spearmanr(grid.med).statistic


def spearman_working(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Spearman between n_a and n_b restricted to configurations that are NOT clipped (score > -1) at BOTH sizes.
    The plain matrix above also ranks the failed configurations among themselves; this one asks the harder question
    "among configurations that work, is the order preserved?". Also returns how many configurations each pair uses.

    Spearman restricted to configurations that are not clipped at both sizes. It asks “among configurations that work, is the order preserved?”.
    It saves a matrix and prints the mean, the minimum, and how many configurations each pair uses. Note that restricting the range lowers correlations somewhat, by construction.
    """
    ok = grid.score > CLIP_LOW
    R, cnt = np.eye(T), np.zeros((T, T), int)
    for a in range(T):
        for b in range(T):
            m = ok[:, a] & ok[:, b]
            cnt[a, b] = m.sum()
            if a != b:
                R[a, b] = spearmanr(grid.med[m, a], grid.med[m, b]).statistic if m.sum() > 30 else np.nan
    return R, cnt


def topfrac_sizes(grid: Grid, frac=0.05) -> list[int]:
    """Size of the 'top 5%' set at each n. Ties at the ceiling (small n) can make it larger than 5% of the grid. Prints the true size of the “top 5%” set at each n, next to the nominal 1,260."""
    return [int((grid.med[:, j] >= np.quantile(grid.med[:, j], 1 - frac)).sum()) for j in range(T)]


def topfrac_jaccard(grid: Grid, frac=0.05, reps=20, seed=0) -> np.ndarray:
    """
    Overlap (Jaccard) of the top-`frac` configurations at two sizes. Each set has EXACTLY round(frac * M) members;
    ties at the cut-off are broken at random and the result is averaged over `reps` draws. (A threshold ">= quantile"
    would put thousands of tied configurations in the set at some n and not at others, and the sets would not be
    comparable.) A low value for a pair can therefore also mean "a large group of configurations is tied at the top".
    """
    rng = np.random.default_rng(seed)
    k = int(round(frac * grid.M))
    J = np.zeros((T, T))
    for _ in range(reps):
        sets = []
        for j in range(T):
            order = np.lexsort((rng.random(grid.M), -grid.med[:, j]))
            m = np.zeros(grid.M, bool)
            m[order[:k]] = True
            sets.append(m)
        for a_ in range(T):
            for b_ in range(T):
                J[a_, b_] += (sets[a_] & sets[b_]).sum() / (sets[a_] | sets[b_]).sum()
    return J / reps


def top_ties(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Per n: (exact, same_level).
      exact      = configurations whose clipped median score equals the best one (to 1e-9);
      same_level = configurations within 1 raw unit of the best median, i.e. at the same attainable nonlinearity level
                   (balanced functions have even nonlinearity; the granular tie-break term is below 1).
    """
    U, B, _ = load_anchors()
    S = grid.score_clip
    best = S.max(axis=0)
    return (S >= best - 1e-9).sum(axis=0), ((best - S) * (U - B) < 1.0).sum(axis=0)


def transfer_regret(grid: Grid) -> np.ndarray:
    """Normalized regret of transferring the best configuration of task s to task t (0 = optimal, 1 = random)."""
    S = grid.score_clip
    R = np.zeros((T, T))
    for s in range(T):
        tied = S[:, s] >= S[:, s].max() - 1e-12            # all configurations tied at the source optimum
        for t in range(T):
            R[s, t] = normalized_regret(S[tied, t].mean(), S[:, t])
    return R


def transfer_regret_levels(grid: Grid) -> np.ndarray:
    """
    Same transfer regret, in ATTAINABLE NONLINEARITY LEVELS instead of a normalised score: (best - chosen) * g_n / 2.
    Balanced functions have even nonlinearity, so 2 raw units = one level. At small n one level is a large part of the
    score range (see the anchor table), so a regret of 'one level' looks big in normalised units but is the smallest
    possible loss.
    """
    U, B, _ = load_anchors()
    gap = U - B
    S = grid.score_clip
    R = np.zeros((T, T))
    for s in range(T):
        tied = S[:, s] >= S[:, s].max() - 1e-12
        for t in range(T):
            R[s, t] = (S[:, t].max() - S[tied, t].mean()) * gap[t] / 2.0
    return R


def importance_tau(shapley: pd.DataFrame) -> np.ndarray:
    """Kendall tau between Shapley importance profiles of different n."""
    A = shapley.to_numpy()
    return np.array([[kendalltau(A[:, a], A[:, b]).statistic for b in range(T)] for a in range(T)])


def warm_start(grid: Grid, src=(5, 9), targets=(10, 12, 13, 14), ks=(1, 10, 100), reps=20, seed=0) -> pd.DataFrame:
    """
    Warm start: evaluate the performance of the best configurations from a source set on a target set.
    Rank configurations by their mean clipped score on n in [src], evaluate the best-k on the target.
    Rank all configurations by their mean score on n = 5 to 9, re-test only the best k on n = 12 to 14, and see how close the winner is to the best configuration there.
    """
    rng = np.random.default_rng(seed)
    cols = [NS.index(n) for n in range(src[0], src[1] + 1)]
    pred = grid.score_clip[:, cols].mean(1)
    rows = []
    for tn in targets:
        if tn <= src[1]:    # a source size cannot also be a target (leakage)
            continue
        y = grid.score_clip[:, NS.index(tn)]
        rho = spearmanr(pred, y).statistic
        for k in ks:
            r = []
            for _ in range(reps):                                    # random tie-breaking
                order = np.lexsort((rng.random(len(pred)), -pred))
                r.append(normalized_regret(y[order[:k]].max(), y))
            rows.append(dict(target=tn, k=k, spearman=rho, regret=np.mean(r)))
    return pd.DataFrame(rows)


SRC_SETS = [(5, 5), (5, 6), (5, 7), (5, 9), (5, 11)]


def warm_start_sources(grid: Grid, targets=(10, 12, 13, 14), ks=(1, 10), **kw) -> pd.DataFrame:
    """Same warm start, but with cheaper and cheaper source sets: how little information is enough? Repeats the warm start with sources of n = 5, 5 to 6, 5 to 7, 5 to 9 and 5 to 11, for k = 1 and 10. This shows how little cheap information is enough, and it appears as a LaTeX table (tab:warmstart_src)."""
    out = []
    for src in SRC_SETS:
        ws = warm_start(grid, src=src, targets=targets, ks=ks, **kw)
        ws['source'] = f'$n\\in[{src[0]},{src[1]}]$'
        out.append(ws)
    return pd.concat(out, ignore_index=True)


def _hm(ax, M, title, cmap, vmin, vmax, fmt='{:.2f}', fontsize=10):
    """Helper function to draw a heatmap with the right labels and colorbar."""
    lab = [str(n) for n in NS]
    im = heatmap(ax, M, lab, lab, cmap=cmap, vmin=vmin, vmax=vmax, fmt=fmt, fontsize=fontsize)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    return im


def fig_transfer_callback(rho, J, R, rw):
    """
    Draw the four panels of the transfer analysis. Here, the size is the number of bits.
    
    Spearman (top left). Rank all 25,200 configurations by their median fitness at size a, and again at size b. Spearman is the correlation between the two rankings: 1 means identical order, 0 means unrelated. Ties get average ranks.
    Top-5% overlap (top right). Take the best 5% of configurations at each size. Jaccard is the size of the overlap divided by the size of the union. It is harsher than Spearman, because it only looks at the best.
    Transfer regret (bottom left). Take the best configuration at the source size, and look up how good it is at the target size. Regret is 0 if it is still the best there, and 1 if it is no better than an average configuration. If several configurations tie for best at the source, the code uses their average at the target, which is the same as picking one at random.
    Spearman among non-clipped configurations (bottom right). The previous panels include all configurations, even those that are clipped at one or both sizes. This panel restricts the Spearman calculation to configurations that are not clipped at either size. It asks “among configurations that work, is the order preserved?”.
    Kendall tau between Shapley importance profiles of different n is not shown, but it is saved as a table, and it shows that the importance of each factor is identical across sizes.
    """
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.6), layout='constrained')
    for a, (M, t, cm, lo, hi) in zip(ax.ravel(), [
            (rho, 'Spearman of configuration ranks', 'cividis', min(0, np.nanmin(rho)), 1),
            (J, f'Top-5{PCT} overlap (Jaccard)', 'cividis', 0, 1),
            (R, 'Regret of transferring the best configuration', 'cividis_r', 0, min(1, np.nanmax(R))),
            (rw, 'Spearman among non-clipped configurations', 'cividis', min(0, np.nanmin(rw)), 1)]):
        im = _hm(a, M, t, cm, lo, hi)
        fig.colorbar(im, ax=a, shrink=0.85, pad=0.01).ax.tick_params(labelsize=8)
    # Spearman, Jaccard and Kendall are symmetric: plain n. Only the regret panel (bottom left) has a direction.
    ax[0, 0].set_ylabel(r'$n$', fontsize=10)
    ax[1, 1].set_xlabel(r'$n$', fontsize=10)
    ax[1, 0].set_ylabel(r'$n$ (source)', fontsize=10)
    ax[1, 0].set_xlabel(r'$n$ (target)', fontsize=10)
    return fig


def run(grid: Grid, importance: dict) -> dict:
    rho, J, R = spearman_matrix(grid), topfrac_jaccard(grid), transfer_regret(grid)
    tau = importance_tau(importance['clip']['shapley'])

    rw, cnt = spearman_working(grid)
    print('Kendall tau between Shapley profiles of different n: min %.2f (1.00 = identical ranking of the 8 knobs)' % np.nanmin(tau))
    
    save_fig(fig_transfer_callback(rho, J, R, rw), 'fig_transfer')
    lab = [f'N{n}' for n in NS]
    for name, M in (('spearman', rho), ('jaccard_top5', J), ('transfer_regret', R), ('shapley_tau', tau)):
        save_table(pd.DataFrame(M, index=lab, columns=lab), f'transfer_{name}')

    Rl = transfer_regret_levels(grid)
    save_table(pd.DataFrame(Rl, index=lab, columns=lab), 'transfer_regret_levels')
    off_ = ~np.eye(T, dtype=bool)

    print('transfer regret in nonlinearity levels: mean %.2f | max %.2f | mean by SOURCE n: %s' % (
        Rl[off_].mean(), Rl.max(), dict(zip(NS, Rl.mean(axis=1).round(2)))))
    exact, same = top_ties(grid)
    print('configurations tied EXACTLY at the best median, per n:', dict(zip(NS, exact.tolist())))
    print('configurations at the SAME nonlinearity level as the best median, per n:', dict(zip(NS, same.tolist())))
    best_cfg = pd.DataFrame({n: grid.cfg.iloc[int(np.argmax(grid.score_clip[:, j]))] for j, n in enumerate(NS)}).T
    best_cfg['score'] = grid.score_clip.max(axis=0)
    best_cfg['n_exact_ties'] = exact
    save_table(best_cfg, 'best_config_per_n')
    print('best configuration per n (first of the tied ones; meaningful only where n_exact_ties is small):\n',
          best_cfg.to_string())

    ws = warm_start(grid)
    save_table(ws.set_index(['target', 'k']), 'transfer_warm_start')
    tab = ws.pivot(index='target', columns='k', values='regret')
    tab.insert(0, 'Spearman', ws.groupby('target').spearman.first())
    tab.columns = ['Spearman'] + [f'best-of-{k}' for k in tab.columns[1:]]
    tab.index = [f'$n={t}$' for t in tab.index]
    print_latex(tab, 'Warm start: configurations ranked by the mean clipped score on $n\\in[5,9]$; normalized regret '
                     'at the target after re-evaluating the best $k$ (0 = optimum, 1 = random configuration).',
                'tab:warmstart', fmt='%.3f')
    print('mean off-diagonal Spearman: %.3f | Spearman(n=5, n=14): %.3f' % (rho[~np.eye(T, dtype=bool)].mean(), rho[0, -1]))

    # harder question: order among configurations that work at both sizes

    off = ~np.eye(T, dtype=bool)
    save_table(pd.DataFrame(rw, index=lab, columns=lab), 'transfer_spearman_working')
    save_table(pd.DataFrame(cnt, index=lab, columns=lab), 'transfer_spearman_working_count')
    print('Spearman among non-clipped configurations: mean off-diagonal %.3f | min %.3f | (n=5, n=14): %.3f | '
          'configurations used per pair: min %d, median %d' % (np.nanmean(rw[off]), np.nanmin(rw[off]), rw[0, -1],
                                                              cnt[off].min(), np.median(cnt[off])))
    print('size of the top-5%% set per n (nominal %d):' % round(0.05 * grid.M), dict(zip(NS, topfrac_sizes(grid))))

    # how cheap can the source be?
    wss = warm_start_sources(grid)
    save_table(wss.set_index(['source', 'target', 'k']), 'transfer_warm_start_sources')
    order = [f'$n\\in[{a},{b}]$' for a, b in SRC_SETS]
    tab2 = wss.pivot(index='source', columns=['target', 'k'], values='regret').loc[order]
    tab2.columns = [f'$n={t}$, $k={k}$' for t, k in tab2.columns]
    print_latex(tab2, 'Warm start with cheaper sources: normalized regret at the target when configurations are ranked by '
                      'the mean clipped score on the listed sizes and the best $k$ are re-evaluated.',
                'tab:warmstart_src', fmt='%.3f')
    return dict(rho=rho, J=J, R=R, tau=tau, warm=ws)
