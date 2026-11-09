"""
Block C - unsupervised layer. Each configuration is a 10-d vector:
  performance : anchored score clipped at -1 (n = 5..14)           -> clusters of equal performance
  stability   : log10(std / gap_n + eps)                            -> clusters of equal run-to-run dispersion
Robustness : same clustering on within-n percentile ranks; subsample stability of k-means.

Each configuration gets two 10-number vectors: its clipped scores at n = 5 to 14 (performance), and its log-dispersion at each n (stability).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score, adjusted_rand_score,
                             adjusted_mutual_info_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from matplotlib.ticker import FuncFormatter
from sklearn.tree import DecisionTreeClassifier, export_text

from gp_common import (FACTORS, NS, OKABE, FACTOR_LABEL, NEG, LEVEL_LABEL, PCT, Grid, present_levels, level_labels, heatmap,
                       save_fig, save_table, print_latex)

SEED = 0


def select_k(X, ks=range(2, 9), sample=6000):
    """
    For each k, fit k-means and report three indices (it does NOT choose k by itself):
      silhouette (higher = better separated; computed on a random subsample of `sample` points),
      Davies-Bouldin (lower = better), inertia (within-cluster sum of squares; look for an elbow).
    """
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=SEED).fit(X)
        rows.append(dict(k=k, silhouette=silhouette_score(X[idx], km.labels_[idx]),
                         davies_bouldin=davies_bouldin_score(X, km.labels_), inertia=km.inertia_))
    return pd.DataFrame(rows).set_index('k')


def fit_ordered(X, k, order_value, ascending=False):
    """
    k-means (best of 20 random starts, i.e. lowest inertia) with clusters renumbered by the mean of `order_value`.
    ascending=False: cluster 0 has the HIGHEST mean order_value (performance: 0 = best).
    ascending=True : cluster 0 has the LOWEST mean order_value (stability: 0 = lowest dispersion = most stable).
    """
    km = KMeans(k, n_init=20, random_state=SEED).fit(X)
    means = np.array([order_value[km.labels_ == c].mean() for c in range(k)])
    order = np.argsort(means if ascending else -means)
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    km.cluster_centers_ = km.cluster_centers_[order]
    km.labels_ = remap[km.labels_]
    return km


def subsampling_stability(X, km, B=20, frac=0.8):
    """
    Subsampling stability (not a classical bootstrap): B times, re-fit k-means on a random `frac` (80%) of the
    configurations, assign ALL configurations to the nearest new centre, and compare with the original labels via ARI.
    Returns mean and std of the ARI over the B repetitions. It tests sensitivity to the data, not to the random seed
    (the seed study is in gp_c2_cluster_robustness.py).
    """
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(B):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        kb = KMeans(km.n_clusters, n_init=5, random_state=int(rng.integers(1e9))).fit(X[idx])
        out.append(adjusted_rand_score(km.labels_, kb.predict(X)))
    return float(np.mean(out)), float(np.std(out))


def enrichment(cfg: pd.DataFrame, labels: np.ndarray):
    """
    log2 lift of each factor level inside each cluster and Cramer's V per factor.
      lift(level, cluster) = P(level | cluster) / P(level in the whole grid); log2 lift = 0 means "as common as expected",
      +1 twice as common, -1 half as common, -inf the level never occurs in the cluster.
      Cramer's V in [0, 1] = strength of the association between the factor and the cluster labels.
    """
    k = labels.max() + 1
    rows, V = [], {}
    for f in FACTORS:
        ct = pd.crosstab(cfg[f], labels).reindex(present_levels(cfg, f))
        chi2 = chi2_contingency(ct.to_numpy())[0]
        V[f] = np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1)))
        lift = (ct / ct.sum(axis=0)).div(ct.sum(axis=1) / ct.to_numpy().sum(), axis=0)
        for l in ct.index:
            with np.errstate(divide='ignore'):
                rows.append((f, l, *np.log2(lift.loc[l].to_numpy(float))))      # -inf where the level never occurs in the cluster
    return pd.DataFrame(rows, columns=['factor', 'level'] + [f'C{c}' for c in range(k)]), V


def rules(cfg, labels, depth=4):
    """
    Extract decision rules from the configuration and labels.
    Returns the text of the tree and the cross-validated balanced accuracy.
    """
    X = pd.get_dummies(cfg[FACTORS]).astype(int)
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=100, class_weight='balanced', random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)      # rows are sorted by factor levels: always shuffle
    acc = cross_val_score(tree, X, labels, cv=cv, scoring='balanced_accuracy')
    tree.fit(X, labels)
    return export_text(tree, feature_names=list(X.columns)), acc


def pareto_front(x, y, mask):
    """
    Maximize x, minimize y, restricted to mask.
    Returns the indices of the Pareto front, sorted by decreasing x.
    """
    idx = np.where(mask)[0]
    o = idx[np.argsort(-x[idx], kind='stable')]
    front, best = [], np.inf
    for i in o:
        if y[i] < best:
            front.append(i)
            best = y[i]
    return np.array(front)


# ------------------------------------------------------------------ plots
def fig_profiles_callback(grid, lp, ls, kp, ks):
    """
    Create a figure with two subplots showing the profiles of the grid points.
    Left: mean and std of the clipped scores for each performance cluster.
    Right: mean and std of the log-dispersion for each stability cluster.
    The legend shows the cluster number and the share of configurations in that cluster.
    """
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), layout='constrained')
    for c in range(kp):
        m = lp == c
        mu, sd = grid.score_clip[m].mean(0), grid.score_clip[m].std(0)
        ax[0].plot(NS, mu, 'o-', ms=3, color=OKABE[c], label=f'P{c} ({100 * m.mean():.0f}{PCT})')
        ax[0].fill_between(NS, mu - sd, mu + sd, color=OKABE[c], alpha=0.15, linewidth=0)
    for c in range(ks):
        m = ls == c
        mu, sd = grid.log_std[m].mean(0), grid.log_std[m].std(0)
        ax[1].plot(NS, mu, 'o-', ms=3, color=OKABE[c], label=f'S{c} ({100 * m.mean():.0f}{PCT})')
        ax[1].fill_between(NS, mu - sd, mu + sd, color=OKABE[c], alpha=0.15, linewidth=0)
    ax[0].set_ylabel('score (clipped)', fontsize=10)
    ax[1].set_ylabel(r'$\log_{10}(\sigma/g_n)$', fontsize=10)
    ax[0].set_title('Performance clusters', fontsize=10)
    ax[1].set_title('Stability clusters', fontsize=10)
    for a in ax:
        a.set_xlabel(r'$n$', fontsize=10)
        a.grid(True, color='gray', linestyle='--', linewidth=0.5)
        a.set_xticks(NS)
        a.legend(fontsize=8, ncol=2)
    return fig


def fig_embedding_callback(grid, lp, ls, max_pts=8000):
    """
    Create a figure with three subplots showing the embedding of the grid points.
    PCA finds the direction in the 10-dimensional space along which the configurations vary most (PC1), and then the next one at right angles (PC2). Each point is placed by its coordinates along these two directions.
    The axis label gives the share of total variance each direction captures.
    A loading is the weight a PC puts on each of the ten scores. The right-hand panel plots those weights against n.
    """
    rng = np.random.default_rng(SEED)
    idx = rng.choice(grid.M, min(max_pts, grid.M), replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), layout='constrained', gridspec_kw={'width_ratios': [1, 1, 0.8]})
    pcs = {}
    for a, (X, lab, title) in zip(ax[:2], [(grid.score_clip, lp, 'PCA of performance'),
                                           (grid.log_std, ls, 'PCA of stability')]):
        p = PCA(2, random_state=SEED).fit(X)
        Z = p.transform(X)
        pcs[title] = p
        for c in np.unique(lab):
            m = idx[lab[idx] == c]
            a.scatter(Z[m, 0], Z[m, 1], s=2, color=OKABE[c], alpha=0.5, rasterized=True, linewidths=0, label=f'{c}')
        a.set_xlabel(f'PC1 ({100 * p.explained_variance_ratio_[0]:.0f}{PCT})', fontsize=10)
        a.set_ylabel(f'PC2 ({100 * p.explained_variance_ratio_[1]:.0f}{PCT})', fontsize=10)
        a.set_title(title, fontsize=10)
        a.grid(True, color='gray', linestyle='--', linewidth=0.5)
        fmt = FuncFormatter(lambda v, _: f'${round(v, 6) + 0.0:g}$')       # explicit math-mode numbers: robust minus sign
        a.xaxis.set_major_formatter(fmt)
        a.yaxis.set_major_formatter(fmt)
        pre = 'P' if title.endswith('performance') else 'S'
        h = [plt.Line2D([0], [0], marker='o', ls='', color=OKABE[c], ms=4, label=f'{pre}{c}') for c in np.unique(lab)]
        a.legend(handles=h, fontsize=8, ncol=2, loc='best')
    p = pcs['PCA of performance']
    for i, (ls_, mk) in enumerate([('-', 'o'), ('--', 's')]):
        ax[2].plot(NS, p.components_[i], ls_, marker=mk, ms=3, color='black', label=f'PC{i + 1}')
    ax[2].axhline(0, color='gray', lw=0.6)
    ax[2].set_xlabel(r'$n$', fontsize=10)
    ax[2].set_title('Performance loadings', fontsize=10)
    ax[2].legend(fontsize=8)
    ax[2].set_xticks(NS[::3])
    return fig


def fig_enrichment_callback(enr, V, k, prefix='P'):
    """
    Plot the enrichment of factors in the clusters.
    Each row is a factor level, each column is a cluster. The colour shows the log2 lift of that level in that cluster.
    The rows are grouped by factor, and the factors are ordered by Cramer's V (strength of association with the cluster labels). The Cramer's V value is printed next to each factor.

    What a cell shows. Take a level, say “Function set: All”, and a cluster, say P3. Two numbers matter:
    - the share of P3 made of All configurations;
    - the share of All in the entire grid (20%, because the grid is balanced).
    Their ratio is the lift. The figure shows its log base 2, so that over- and under-representation are symmetric:

    0 means the level is as common in the cluster as in the grid;
    +1 means twice as common, -1 half as common, -2 a quarter as common;
    -∞ means the level never occurs in that cluster.

    A level with zero members in a cluster has a lift of 0, and the log of 0 does not exist. These are the strongest possible negative results, such as “this function set never ends up in this cluster”.
    They show “−∞”. They are plotted with the darkest blue of the scale, the same as the strongest finite value.

    Cramér's V (the number on the right). One value per knob, between 0 and 1. It measures how strongly knowing the level of that knob tells you which cluster the configuration is in.
    Specifically, it is computed between two equal-sized categorical variables: the knob's levels and the cluster labels (because a configuration belongs to one cluster, and that configuration has a given level for the knob/factor under analysis).

    0 means the knob's levels are spread over the clusters exactly like the grid; 1 means the knob determines the cluster.
    Rough guide: about 0.1 is weak, 0.3 moderate, 0.5 or more strong. For tables with more levels the cutoffs are somewhat lower.
    It is computed from the chi-square statistic of the knob-by-cluster table, divided by what would be the maximum for that table size, so it does not grow with the number of configurations. No p-values are reported, because with 25,200 configurations every association would be “significant”.
    It answers a different question from Shapley: V is about which cluster a configuration falls in, while Shapley is about how much the numerical score varies.
    """
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 7.2), layout='constrained')
    cols = [f'C{c}' for c in range(k)]
    M = enr[cols].to_numpy(float)
    ylab = [f'{FACTOR_LABEL[f]}: ' + level_labels(f, [l])[0] for f, l in zip(enr.factor, enr.level)]
    empty = np.isneginf(M)                                   # level never occurs in this cluster
    lim = np.abs(M[np.isfinite(M)]).max()
    M = np.where(empty, -lim, M)                             # drawn with the most negative colour
    annot = [[r'$-\infty$' if empty[i, j] else f'{(M[i, j] + 0.0):.1f}'.replace('-0.0', '0.0') for j in range(k)]
             for i in range(M.shape[0])]
    im = heatmap(ax, M, [f'{prefix}{c}' for c in range(k)], ylab, cmap='RdBu_r', vmin=-lim, vmax=lim, annot=annot, fontsize=10)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    bounds = np.cumsum([enr.factor.eq(f).sum() for f in FACTORS])[:-1] - 0.5
    for b in bounds:
        ax.axhline(b, color='black', lw=0.8)
    start = 0
    for f in FACTORS:
        n_l = enr.factor.eq(f).sum()
        ax.text(1.02, start + (n_l - 1) / 2, f'$V={V[f]:.2f}$', transform=ax.get_yaxis_transform(),
                fontsize=10, va='center')
        start += n_l
    cb = fig.colorbar(im, ax=ax, location='bottom', shrink=0.9, pad=0.01)
    cb.set_label(r'$\log_2$ lift of level in cluster', fontsize=8)
    cb.ax.tick_params(labelsize=8)
    return fig


def fig_joint_callback(grid, lp, ls, kp, ks, front, ami):
    """
    Create a figure with two subplots showing the joint distribution of performance and stability.
    Left: scatter of the mean score vs mean log-dispersion for each configuration, coloured by performance cluster.
    Right: heatmap of the share of each performance cluster in each stability cluster, with the AMI (adjusted mutual information) between the two clusterings in the title.
    The Pareto front is highlighted in the scatter plot. The AMI measures how much knowing the performance cluster tells you about the stability cluster, and vice versa. It is 0 if the two clusterings are independent, and 1 if they are identical.

    The question: which configurations have no rival that is both better on mean score and more stable? Higher mean score and lower mean log-dispersion are both good, so the front lies at the lower right of the scatter plot.
    The algorithm: keep only configurations with mean score above 0. Sort them from the highest score down. Walk through the list, and keep a configuration only if its dispersion is lower than that of every configuration seen before it.
    Why the filter at 0: failed configurations with zero dispersion would otherwise sit on the front, being very stable but useless.
    A caveat: both axes are averages over n, so the front hides differences between problem sizes. Non-dominated does not mean best. The choice along the front is a preference.
    """
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), layout='constrained', gridspec_kw={'width_ratios': [1.35, 1]})
    x, y = grid.score_clip.mean(1), grid.log_std.mean(1)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(grid.M, min(10000, grid.M), replace=False)
    for c in range(kp):
        m = idx[lp[idx] == c]
        ax[0].scatter(x[m], y[m], s=2, color=OKABE[c], alpha=0.5, rasterized=True, linewidths=0, label=f'P{c}')
    f = front[np.argsort(x[front])]
    ax[0].plot(x[f], y[f], '-', color='black', lw=1.2, label='Pareto front')
    ax[0].scatter(x[f], y[f], s=12, color='black', zorder=3)
    ax[0].set_xlabel(r'mean score over $n$ (higher = better)', fontsize=10)
    ax[0].set_ylabel(r'mean $\log_{10}(\sigma/g_n)$ (lower = more stable)', fontsize=10)
    ax[0].grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=8, markerscale=3, ncol=2)
    ct = pd.crosstab(lp, ls).reindex(index=range(kp), columns=range(ks), fill_value=0)
    P = ct.div(ct.sum(axis=1), axis=0).to_numpy() * 100
    annot = [[f'{P[i, j]:.0f}' for j in range(ks)] for i in range(kp)]
    im = heatmap(ax[1], P, [f'S{j}' for j in range(ks)], [f'P{i}' for i in range(kp)], cmap='cividis', vmin=0, vmax=100,
                 annot=annot, fontsize=10)
    ax[1].set_title(f'Row share ({PCT}); AMI $={ami:.2f}$', fontsize=10)
    return fig


# ------------------------------------------------------------------ driver
def run(grid: Grid, k_perf: int | None = None, k_stab: int | None = None) -> dict:
    Xp, Xs = grid.score_clip, grid.log_std
    sel_p, sel_s = select_k(Xp), select_k(Xs)
    print('k selection (performance):\n', sel_p.round(3).to_string())
    print('k selection (stability):\n', sel_s.round(3).to_string())
    save_table(sel_p, 'kselect_perf')
    save_table(sel_s, 'kselect_stab')
    kp = k_perf or int(sel_p.silhouette.idxmax())
    ks = k_stab or int(sel_s.silhouette.idxmax())
    print(f'using k_perf={kp}, k_stab={ks}  (override with run(grid, k_perf=..., k_stab=...))')

    kmp = fit_ordered(Xp, kp, Xp.mean(1)) # 0 = best, k_perf - 1 = worst
    kms = fit_ordered(Xs, ks, Xs.mean(1), ascending=True) # 0 = most stable, ks - 1 = least stable
    lp, ls = kmp.labels_, kms.labels_
    print('subsampling ARI perf: %.3f +- %.3f | stab: %.3f +- %.3f' % (*subsampling_stability(Xp, kmp), *subsampling_stability(Xs, kms)))
    kmr = fit_ordered(grid.rank, kp, grid.rank.mean(1))
    print('ARI(score-based, rank-based performance clusters) = %.3f' % adjusted_rand_score(lp, kmr.labels_))
    ami = adjusted_mutual_info_score(lp, ls)
    print('AMI(performance, stability clusters) = %.3f' % ami)

    enr_p, Vp = enrichment(grid.cfg, lp)
    enr_s, Vs = enrichment(grid.cfg, ls)
    print("Cramer's V (performance clusters):", {FACTOR_LABEL[f]: round(v, 2) for f, v in Vp.items()})
    print("Cramer's V (stability clusters):  ", {FACTOR_LABEL[f]: round(v, 2) for f, v in Vs.items()})
    for name, lab in (('performance', lp), ('stability', ls)):
        txt, acc = rules(grid.cfg, lab)
        print(f'\nDecision-tree rules for {name} clusters (depth 4, CV balanced acc = {acc.mean():.2f}):\n{txt}')

    m = grid.score_clip.mean(1)
    front = pareto_front(m, Xs.mean(1), mask=m > 0)
    print('\nPareto-front configurations (mean score > 0):')
    print(pd.concat([grid.cfg.iloc[front].reset_index(drop=True),
                     pd.DataFrame({'mean_score': m[front], 'mean_log_std': Xs.mean(1)[front]})], axis=1).round(3).to_string())

    save_fig(fig_profiles_callback(grid, lp, ls, kp, ks), 'fig_cluster_profiles')
    save_fig(fig_embedding_callback(grid, lp, ls), 'fig_cluster_embedding')
    save_fig(fig_enrichment_callback(enr_p, Vp, kp), 'fig_cluster_enrichment_perf')
    save_fig(fig_enrichment_callback(enr_s, Vs, ks, 'S'), 'fig_cluster_enrichment_stab')
    save_fig(fig_joint_callback(grid, lp, ls, kp, ks, front, ami), 'fig_perf_vs_stability')

    summ = pd.DataFrame({
        'size (\\%)': [100 * (lp == c).mean() for c in range(kp)],
        '$n{=}8$': [grid.score_clip[lp == c][:, NS.index(8)].mean() for c in range(kp)],
        '$n{=}11$': [grid.score_clip[lp == c][:, NS.index(11)].mean() for c in range(kp)],
        '$n{=}14$': [grid.score_clip[lp == c][:, NS.index(14)].mean() for c in range(kp)],
        'plateau (\\%)': [100 * grid.plateau[lp == c].mean() for c in range(kp)]},
        index=[f'P{c}' for c in range(kp)])
    print_latex(summ, 'Performance clusters: size, mean clipped score and share of failure-plateau cells.',
                'tab:clusters', fmt='%.2f')
    out = pd.concat([grid.cfg, pd.DataFrame({'perf_cluster': lp, 'stab_cluster': ls})], axis=1)
    save_table(out.set_index(FACTORS), 'cluster_labels')
    return dict(perf=lp, stab=ls)
