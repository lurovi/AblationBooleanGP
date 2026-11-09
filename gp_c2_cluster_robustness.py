"""
Block C2/G - robustness of the clustering.
 (1) seed sensitivity of k-means: N_SEEDS runs with a single random initialisation (n_init=1, exposes local optima)
     and with the production setting (n_init=20). Reference partition = the run with the LOWEST inertia
     (the objective k-means minimises); every other run is compared with it through the ARI.
 (2) other algorithms on the same features: k-means on percentile ranks, Gaussian mixture (diagonal covariance, BIC),
     HDBSCAN (number of clusters found automatically, noise points allowed; main parameter = min_cluster_size). The same performance features are clustered with k-means on percentile ranks, a Gaussian mixture (GMM), and HDBSCAN. Each is compared with k-means.
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI, adjusted_mutual_info_score as AMI, silhouette_score, homogeneity_score, completeness_score

from gp_common import Grid, OKABE, PCT, heatmap, save_fig, save_table, print_latex
import gp_c_clustering as C

try:
    from sklearn.cluster import HDBSCAN
except ImportError:                      # scikit-learn < 1.3
    HDBSCAN = None

warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
N_SEEDS = int(os.environ.get('GP_SEEDS', 50))
HDB_FRACS = (0.005, 0.01, 0.02)          # min_cluster_size as a fraction of the grid


def relabel(lab, value, ascending=False):
    """Renumber clusters so that 0 = best (highest mean `value`, or lowest if ascending)."""
    k = lab.max() + 1
    means = np.array([value[lab == c].mean() for c in range(k)])
    order = np.argsort(means if ascending else -means)
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[lab]


def seed_study(X, k, n_init, n_seeds=N_SEEDS, sample=6000):
    """
    Run k-means with `n_seeds` random seeds, each with `n_init` initialisations, and return a DataFrame with the
    inertia and ARI of each run against the best one (lowest inertia). Also return the labels of the best run, relabelled so that cluster 0 = highest mean value of X.mean(1).

    K-means starts from random centres, and a bad start can end in a worse solution.
    n_init=1 means a single random start. n_init=20 means 20 starts, keeping the best (lowest inertia).
    50 different seeds are used for each setting.
    Each run is compared with the best run using ARI, the score that is 1 for identical groupings and about 0 for random ones.
    The table reports how many seeds reach the best result, the worst ARI, and the silhouette range.
    """
    idx = np.random.default_rng(0).choice(len(X), min(sample, len(X)), replace=False)
    runs = []
    for s in range(n_seeds):
        km = KMeans(k, n_init=n_init, init='k-means++', random_state=s).fit(X)
        runs.append((km.inertia_, km.labels_.copy()))
    b = int(np.argmin([r[0] for r in runs]))
    best_inertia, best_lab = runs[b]
    df = pd.DataFrame([dict(seed=s, inertia=i, gap_pct=100 * (i / best_inertia - 1), ari=ARI(best_lab, l),
                            silhouette=silhouette_score(X[idx], l[idx])) for s, (i, l) in enumerate(runs)])
    return df, best_lab


def gmm_scan(X, ks=range(2, 9)):
    """
    Scan the BIC for a range of k values for a Gaussian Mixture Model.
    """
    rows = []
    for k in ks:
        g = GaussianMixture(k, covariance_type='diag', reg_covar=1e-3, n_init=5, random_state=0).fit(X)
        rows.append(dict(k=k, bic=g.bic(X)))
    return pd.DataFrame(rows).set_index('k')


def bic_message(gm):
    """
    Generate a message indicating the BIC-minimising k value. The BIC is a score for how well a mixture with k components fits, penalising complexity. Lower is better.
    """
    k = int(gm.bic.idxmin())
    edge = k in (gm.index.min(), gm.index.max())
    return f'BIC-minimising k = {k}' + (' (edge of the scanned range: BIC keeps improving, so it does not identify a k)' if edge else '')


def k_sensitivity(X, kp, ks=None):
    """
    How the k-means partition changes with k: ARI between k and k+1, and whether the k+1 clusters are nested in
    the k clusters (homogeneity = 1 means every cluster of k+1 lies inside a single cluster of k)."""
    ks = sorted(set(ks or range(2, 7)) | {max(2, kp - 1), kp, kp + 1})
    L = {k: C.fit_ordered(X, k, X.mean(1)).labels_ for k in ks}
    rows = [{'step': f'{k} to {k + 1}', 'ARI': ARI(L[k], L[k + 1]), 'nested (homogeneity)': homogeneity_score(L[k], L[k + 1])}
            for k in ks[:-1]]
    tab = pd.DataFrame(rows).set_index('step')
    print('k sensitivity (performance features):\n', tab.round(3).to_string())
    if kp - 1 in L and kp - 1 >= 2:
        print(f'\nk={kp - 1} (rows) vs k={kp} (columns):\n', pd.crosstab(L[kp - 1], L[kp]).to_string())
    return tab


def hdbscan_scan(X, ref):
    """
    Scan the HDBSCAN algorithm for a range of minimum cluster sizes. It finds dense regions by itself and marks sparse points as noise. min_cluster_size is the smallest group it will report, here 0.5%, 1% and 2% of the grid.
    """
    out, labs = [], {}
    if HDBSCAN is None:
        print('HDBSCAN not available (needs scikit-learn >= 1.3): skipped')
        return pd.DataFrame(), labs
    for f in HDB_FRACS:
        lab = HDBSCAN(min_cluster_size=max(20, int(f * len(X)))).fit_predict(X)
        labs[f] = lab
        m = lab >= 0
        out.append(dict(min_cluster_frac_pct=100 * f, n_clusters=int(lab.max() + 1), noise_pct=100 * (~m).mean(),
                        AMI_vs_kmeans=AMI(ref, lab),
                        homogeneity_non_noise=homogeneity_score(ref[m], lab[m]) if m.any() else np.nan,
                        completeness_non_noise=completeness_score(ref[m], lab[m]) if m.any() else np.nan))
    return pd.DataFrame(out), labs


def fig_callback(seed_df, ari_mat, names):
    """
    Create a figure with (1) the ARI of each random seed against the best run, and (2) a heatmap of the ARI between algorithms.
    """
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), layout='constrained', gridspec_kw={'width_ratios': [1.1, 1]})
    for (lab, df), c, mk in zip(seed_df.items(), OKABE, ['o', 's', '^', 'D']):
        jitter = np.random.default_rng(1).normal(0, 0.002, len(df))
        ax[0].scatter(df.gap_pct, df.ari + jitter, s=14, color=c, marker=mk, alpha=0.7, label=lab)
    ax[0].set_xlabel(f'inertia above the best run ({PCT})', fontsize=10)
    ax[0].set_ylabel('ARI with the best run', fontsize=10)
    ax[0].set_ylim(min(0.9, ax[0].get_ylim()[0]), 1.01)
    ax[0].grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=8)
    ax[0].set_title(f'{N_SEEDS} random seeds', fontsize=10)
    ax[0].tick_params(axis='both', labelsize=10)
    im = heatmap(ax[1], ari_mat, names, names, cmap='cividis', vmin=0, vmax=1, fmt='{:.2f}', fontsize=10)
    ax[1].tick_params(axis='x', labelsize=10)
    plt.setp(ax[1].get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
    ax[1].tick_params(axis='y', labelsize=10)
    ax[1].set_title('ARI between algorithms (performance)', fontsize=10)
    fig.colorbar(im, ax=ax[1], shrink=0.85, pad=0.01).ax.tick_params(labelsize=10)
    return fig


def run(grid: Grid, k_perf: int | None = None, k_stab: int | None = None) -> None:
    Xp, Xs = grid.score_clip, grid.log_std
    kp = k_perf or int(C.select_k(Xp).silhouette.idxmax())
    ks = k_stab or int(C.select_k(Xs).silhouette.idxmax())
    print(f'k_perf={kp}, k_stab={ks}')
    rows, seed_dfs, best = [], {}, {}
    for feat, X, k in (('performance', Xp, kp), ('stability', Xs, ks)):
        for n_init in (1, 20):
            df, lab = seed_study(X, k, n_init)
            seed_dfs[f'{feat[:4]}, n_init={n_init}'] = df
            if n_init == 20:
                best[feat] = relabel(lab, X.mean(1), ascending=(feat == 'stability'))
            rows.append({'features': feat, 'n_init': n_init,
                         'seeds within 0.01\\% of best inertia (\\%)': f'{100 * (df.gap_pct < 1e-2).mean():.0f}',
                         'min ARI vs best': f'{df.ari.min():.3f}',
                         'silhouette median [min, max]': f'{df.silhouette.median():.3f} [{df.silhouette.min():.3f}, {df.silhouette.max():.3f}]'})
            save_table(df.set_index('seed'), f'seeds_{feat}_ninit{n_init}')
    tab = pd.DataFrame(rows).set_index(['features', 'n_init'])
    print_latex(tab, f'Sensitivity of $k$-means to the random seed ({N_SEEDS} seeds; reference = lowest-inertia run).',
                'tab:seeds', fmt='%s', column_format='llrrl')

    # other algorithms, performance features
    ref = best['performance']
    labs = {'k-means': ref,
            'k-means (ranks)': KMeans(kp, n_init=20, random_state=0).fit_predict(grid.rank),
            f'GMM (k={kp})': GaussianMixture(kp, covariance_type='diag', reg_covar=1e-3, n_init=5, random_state=0).fit_predict(Xp)}
    gm = gmm_scan(Xp)
    print('GMM BIC (performance):\n', gm.round(0).to_string(), '\n' + bic_message(gm))
    hd, hl = hdbscan_scan(Xp, ref)
    print('HDBSCAN (performance):\n', hd.round(3).to_string(index=False))
    # HDBSCAN is NOT in the ARI matrix: it returns many small clusters plus noise, so its ARI against a 4-cluster
    # partition is low by construction even when it only refines it. It is compared through homogeneity/completeness.
    names = list(labs)
    M = np.array([[ARI(labs[a], labs[b]) for b in names] for a in names])
    save_table(pd.DataFrame(M, index=names, columns=names), 'cluster_method_ari_perf')
    print('ARI between algorithms (performance):\n', pd.DataFrame(M, index=names, columns=names).round(2).to_string())
    if hl:
        print('\nk-means (rows) vs HDBSCAN 1% (columns; -1 = noise):\n', pd.crosstab(ref, hl[0.01]).to_string())
    save_table(k_sensitivity(Xp, kp), 'k_sensitivity_perf')

    # stability features
    hs, hls = hdbscan_scan(Xs, best['stability'])
    print('\nHDBSCAN (stability):\n', hs.round(3).to_string(index=False))
    if hls:
        for f in HDB_FRACS:
            print(f'\nk-means stability clusters (rows) vs HDBSCAN {100 * f:g}% (columns; -1 = noise):\n',
                  pd.crosstab(best['stability'], hls[f]).to_string())
    gs = gmm_scan(Xs)
    print('GMM BIC (stability):', bic_message(gs))
    gms = GaussianMixture(ks, covariance_type='diag', reg_covar=1e-3, n_init=5, random_state=0).fit_predict(Xs)
    print(f'ARI(GMM k={ks}, k-means) on stability features = {ARI(gms, best["stability"]):.3f}')
    save_fig(fig_callback(seed_dfs, M, names), 'fig_cluster_robustness')
