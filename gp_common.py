"""
Shared infrastructure: paths, plotting style, loading/validation, scoring, tensors, statistics.

Environment variables
  GP_DATA_DIR : folder with the CSV files            (default: current dir)
  GP_OUT_DIR  : figures folder                       (default: img)
  GP_NO_TEX=1 : disable LaTeX rendering (smoke tests without a TeX install)
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
import matplotlib.pyplot as plt # Matplotlib 3.11.1+ for the minus sign bug in tick labels rendered in pdf by latex (https://github.com/matplotlib/matplotlib/issues/18405)
import matplotlib.ticker as ticker

# ... your heatmap / plot code ...

# Force tick labels into LaTeX math blocks ($ -x $) for both axes
#plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter(r'$%g$'))
#plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter(r'$%g$'))

#plt.savefig(IMG_DIR / 'your_plot.pdf', dpi=DPI, bbox_inches='tight')
#(If your axes are on a log scale, use ticker.LogFormatterMathtext() instead).

# --------------------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get('GP_DATA_DIR', '.'))
IMG_DIR = Path(os.environ.get('GP_OUT_DIR', 'img'))
TAB_DIR = Path('tables')
RES_DIR = Path('results')
DPI = 800

NS = list(range(5, 15)) # the 10 n_bits values representing the problem instances
NCOLS = [f'N{n}' for n in NS] # column names for the 10 N values in the CSVs
FACTORS = ['TreeInit', 'PopGenRatio', 'CrossMutRate', 'Crossover', 'Mutation', 'FunctionSet', 'TournamentSize', 'Elites'] # factor names in the CSVs, basically the features (hyperparameters/knobs) of the GP configuration
MIXED = ['EqualCrossoverMutation', 'HighCrossoverLowMutation', 'LowCrossoverHighMutation'] # only main variant has these 3 mixed rates, dedup variant has 5 rates (including OnlyCrossover and OnlyMutation)
ALL_RATES = MIXED + ['OnlyCrossover', 'OnlyMutation']
NA_TOKEN = 'NotUsed'
CLIP_LOW = -1.0          # scores below -1 (worse than the random-balanced-to-best gap) are clipped
STD_EPS = 1e-3           # floor added to std/gap before log10 (differences < 1e-3 of the gap come only from the granular tie-break term)
PLATEAU_TOL = 1e-3       # |median - (1 - 2^-n)| < tol  -> "balanced linear plateau"

# The order of levels in the CSVs is not guaranteed, so we define a canonical order for each factor. The plotting code will use this order to ensure consistent colors and positions across plots. Levels are the distinct values of each factor in the CSVs, e.g., TreeInit has 3 levels: Full, Grow, HalfAndHalf. The order of levels is used for plotting and for paired comparisons.
LEVEL_ORDER = {
    'TreeInit': ['Full', 'Grow', 'HalfAndHalf'],
    'PopGenRatio': ['0.01', '0.04', '0.25', '1.0', '4.0', '25.0', '100.0'],
    'CrossMutRate': ['LowCrossoverHighMutation', 'EqualCrossoverMutation',
                     'HighCrossoverLowMutation', 'OnlyCrossover', 'OnlyMutation'],
    'Crossover': ['CxOnePoint', 'CxOnePointLeafBiased', NA_TOKEN],
    'Mutation': ['MutUniform', 'MutInsert', 'MutNodeReplacement', 'MutShrink'],
    'FunctionSet': ['Minimal', 'Basic', 'BasicWithIf', 'Custom', 'All'],
    'TournamentSize': ['2', '3', '4', '5', '6'],
    'Elites': ['1', '5'],
}

# --------------------------------------------------------------------------------------
# Plot style (same rcParams as the author's guidelines) and labels
# --------------------------------------------------------------------------------------
USE_TEX = os.environ.get('GP_NO_TEX', '0') != '1'
PCT = r'\%' if USE_TEX else '%'
DAG = r'$\dagger$'
AST = r'$*$'
NEG = r'-'  # en dash for negative numbers in PDF output (see setup_style())

# Okabe-Ito color palette (DavidMathLogic, CC-BY-SA 4.0) for colorblind-safe plots. The palette is designed to be distinguishable by people with various types of color vision deficiencies. The colors are in hex format and can be used directly in matplotlib.
OKABE = ['#0072B2', '#F0E442', '#009E73', '#D55E00', '#E69F00', '#3E3535', '#56B4E9', '#CC79A7']


def setup_style() -> None:
    preamble = r'''
            \usepackage{amsmath}
            \usepackage{libertine}
            \usepackage{xspace}
            '''
    rc = {'text.usetex': USE_TEX,
          'font.family': 'serif',
          'font.serif': 'Computer Modern Roman' if USE_TEX else 'DejaVu Serif',
          'pdf.fonttype': 42, 'ps.fonttype': 42,
          'axes.formatter.use_mathtext': True,
          'axes.unicode_minus': False # try to change to True if you have issues and a TeX install and want the minus sign to be rendered as a proper minus sign in PDF output
          }
    if USE_TEX:
        rc['text.latex.preamble'] = preamble
    plt.rcParams.update(rc)

    # With usetex, some matplotlib versions subset the embedded TeX fonts and DROP the math minus (glyph code 0 of
    # the cmsy font): negative tick labels then print as positive numbers in the PDF. Typeset the sign of tick labels
    # with the text font instead (an en dash, glyph code != 0). Also used by NEG for hand-made labels.
    # from matplotlib import ticker
    # ticker.Formatter.fix_minus = staticmethod(lambda s: s.replace('-', NEG).replace('\N{MINUS SIGN}', NEG))

setup_style()

FACTOR_LABEL = {
    'TreeInit': 'Tree init.', 'PopGenRatio': 'Pop./Gen.', 'CrossMutRate': 'Cx/Mut rate',
    'Crossover': 'Crossover', 'Mutation': 'Mutation', 'FunctionSet': 'Function set',
    'TournamentSize': 'Tournament', 'Elites': 'Elites',
}
_X = r'{\times}' if USE_TEX else 'x'
LEVEL_LABEL = {
    'TreeInit': {'Full': 'Full', 'Grow': 'Grow', 'HalfAndHalf': 'Half\\&Half' if USE_TEX else 'Half&Half'},
    'PopGenRatio': {'0.01': f'$10{_X}1000$', '0.04': f'$20{_X}500$', '0.25': f'$50{_X}200$',
                    '1.0': f'$100{_X}100$', '4.0': f'$200{_X}50$', '25.0': f'$500{_X}20$',
                    '100.0': f'$1000{_X}10$'},
    'CrossMutRate': {'LowCrossoverHighMutation': '0.3/0.7', 'EqualCrossoverMutation': '0.5/0.5',
                     'HighCrossoverLowMutation': '0.7/0.3', 'OnlyCrossover': '1.0/0.0',
                     'OnlyMutation': '0.0/1.0'},
    'Crossover': {'CxOnePoint': 'One-point', 'CxOnePointLeafBiased': 'Leaf-biased', NA_TOKEN: 'n/a'},
    'Mutation': {'MutUniform': 'Uniform', 'MutInsert': 'Insert', 'MutNodeReplacement': 'Node repl.',
                 'MutShrink': 'Shrink'},
    'FunctionSet': {'Minimal': 'Minimal', 'Basic': 'Basic', 'BasicWithIf': 'Basic+if',
                    'Custom': 'Custom', 'All': 'All'},
    'TournamentSize': {k: k for k in LEVEL_ORDER['TournamentSize']},
    'Elites': {'1': '1', '5': '5'},
}


def level_labels(factor: str, levels: list[str]) -> list[str]:
    """Return the human-readable labels for the levels of a factor, in the order given by `levels`."""
    return [LEVEL_LABEL[factor].get(l, l) for l in levels]


def n_colors():
    """Return a dict mapping each N to a color from the viridis colormap, evenly spaced."""
    cmap = plt.get_cmap('viridis')
    return {n: cmap(i / (len(NS) - 1)) for i, n in enumerate(NS)}


def save_fig(fig, name: str, dpi: int = DPI) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_DIR / f'{name}.png', dpi=dpi)
    fig.savefig(IMG_DIR / f'{name}.pdf', dpi=dpi)
    fig.clf()
    plt.cla()
    plt.close(fig)


def save_table(df: pd.DataFrame, name: str) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES_DIR / f'{name}.csv')


def print_latex(df: pd.DataFrame, caption: str, label: str, fmt: str = '%.2f', index: bool = True, column_format: str | None = None) -> str:
    """Print (and store in tables/) a booktabs table ready to be pasted in the paper."""
    body = df.to_latex(index=index, float_format=lambda x: fmt % x, escape=False,
                       column_format=column_format, na_rep='--')
    tex = ('\\begin{table}[t]\n\\centering\n\\small\n'
           f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
           '\\resizebox{\\columnwidth}{!}{%\n' + body.replace('\\begin{tabular}', '\\begin{tabular}', 1).rstrip() +
           '}\n\\end{table}\n')
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    (TAB_DIR / f'{label.replace(":", "_")}.tex').write_text(tex)
    print(f'\n% ---- {label} ----\n{tex}')
    return tex


# --------------------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------------------

# Mapping from kind to CSV file name (without the variant suffix, which is either main or dedup, and the .csv extension). The CSVs are generated by the generation script.
_FILES = {'med': 'gp_bool_data_med_pivot', 'std': 'gp_bool_data_std_pivot', 'med_scaled': 'gp_bool_data_med_scaled_pivot', 'std_scaled': 'gp_bool_data_std_scaled_pivot'}


def _read(kind: str, variant: str) -> pd.DataFrame:
    """Read a CSV file and validate its contents. Raises AssertionError if the file is missing or malformed."""
    path = DATA_DIR / f'{_FILES[kind]}_{variant}.csv'
    df = pd.read_csv(path, dtype={k: str for k in FACTORS}, keep_default_na=False)
    assert list(df.columns) == FACTORS + NCOLS, f'{path}: unexpected columns {list(df.columns)}'
    assert not df[NCOLS].isna().any().any(), f'{path}: NaN in values'
    assert not df.duplicated(subset=FACTORS).any(), f'{path}: duplicated configurations'
    return df.sort_values(FACTORS).reset_index(drop=True)


def load_anchors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the upper bounds and anchor values for each N. Returns (U, B, E) as 1D arrays of length len(NS)."""
    U = pd.read_csv(DATA_DIR / 'upper_bounds.csv', index_col=0)['U'].reindex(NS).to_numpy(float)
    B = pd.read_csv(DATA_DIR / 'anchors.csv', index_col=0)['B'].reindex(NS).to_numpy(float)
    E = pd.read_csv(DATA_DIR / 'anchor_errors.csv', index_col=0)['B_err'].reindex(NS).to_numpy(float)
    assert not (np.isnan(U).any() or np.isnan(B).any())
    return U, B, E


@dataclass
class Grid:
    """A grid of configurations and their associated data (M configurations, 10 N values each with the respective scores)."""
    cfg: pd.DataFrame        # (M, 8) factor levels as strings
    med: np.ndarray          # (M, 10) raw median granular fitness
    std: np.ndarray          # (M, 10) raw std (ddof=1)
    score: np.ndarray        # (M, 10) anchored score (f - b_n) / (u_n - b_n), unclipped
    score_clip: np.ndarray   # (M, 10) same, clipped below at CLIP_LOW
    std_score: np.ndarray    # (M, 10) std / (u_n - b_n)
    log_std: np.ndarray      # (M, 10) log10(std_score + eps)
    rank: np.ndarray         # (M, 10) within-n percentile rank of the median, in (0, 1] -> 1 = best, 0.5 = median, 0 = worst, it is not a score, but a measure of relative performance, i.e., the rank of the median among all configurations for that N, divided by the number of configurations (M)
    plateau: np.ndarray      # (M, 10) bool, MEDIAN run ended on a balanced linear function (>= ~half of the 20 runs)
    eps: float               # small value added to std_score before log10 to avoid -inf for very small stds (see STD_EPS)

    @property
    def M(self) -> int:
        """Number of configurations in the grid."""
        return len(self.cfg)


def load_grid(variant: str = 'main') -> Grid:
    """variant: 'main' (3 mixed rates, 25,200 cfgs) or 'dedup' (5 rates, Crossover=NotUsed under OnlyMutation)."""
    D = {k: _read(k, variant) for k in _FILES}
    for k, d in D.items():
        assert d[FACTORS].equals(D['med'][FACTORS]), f'{k}: configuration keys differ from med file'
    if variant == 'main':
        assert len(D['med']) == 25200 and set(D['med'].CrossMutRate) == set(MIXED)
    else:
        assert len(D['med']) == 37800 and set(D['med'].CrossMutRate) == set(ALL_RATES)
        assert (D['med'].loc[D['med'].CrossMutRate == 'OnlyMutation', 'Crossover'] == NA_TOKEN).all()
    cfg = D['med'][FACTORS].copy()
    med, std = D['med'][NCOLS].to_numpy(float), D['std'][NCOLS].to_numpy(float)
    U, B, _ = load_anchors()
    gap = U - B
    score, std_score = (med - B) / gap, std / gap
    # consistency with the scaled CSVs written by the generation script
    assert np.allclose(score, D['med_scaled'][NCOLS].to_numpy(float), atol=1e-7), 'median scaling mismatch'
    assert np.allclose(std_score, D['std_scaled'][NCOLS].to_numpy(float), atol=1e-7), 'std scaling mismatch'
    eps = STD_EPS      # dispersions far below one nonlinearity step are treated as ~0 (see STD_EPS)
    rank = np.column_stack([rankdata(med[:, j], method='average') / len(med) for j in range(len(NS))])
    lin = np.array([1 - 2.0 ** -n for n in NS])
    return Grid(cfg=cfg, med=med, std=std, score=score, score_clip=np.maximum(score, CLIP_LOW),
                std_score=std_score, log_std=np.log10(std_score + eps), rank=rank,
                plateau=np.abs(med - lin) < PLATEAU_TOL, eps=eps)


# --------------------------------------------------------------------------------------
# Tensors (complete product grids)
# --------------------------------------------------------------------------------------
def present_levels(cfg: pd.DataFrame, factor: str) -> list[str]:
    """Return the levels of a factor that are present in the grid, in the canonical order defined by LEVEL_ORDER."""
    s = set(cfg[factor])
    lv = [l for l in LEVEL_ORDER[factor] if l in s]
    assert len(lv) == len(s), f'{factor}: unknown levels {s - set(lv)}'
    return lv


def to_tensor(cfg: pd.DataFrame, V: np.ndarray, factors: list[str]) -> tuple[np.ndarray, list[list[str]]]:
    """
    Reshape (M, T) values, with M being the number of configurations and T being the number of NBITS (N) values, to a (d_1, ..., d_F, T) tensor, with F being the number of factors and d_j being the number of levels for each factor.
    Asserts that the grid is a complete product. V is a matrix of values (e.g., scores) for each configuration and each N.
    The function computes the levels of each factor, checks that the number of configurations matches the product of the number of levels, and then reshapes V into a tensor where each axis corresponds to a factor and the last axis corresponds to the N values.
    The resulting tensor can be used for marginalization and plotting.
    Returns (Y, levels) where Y is the tensor and levels is a list of lists of levels for each factor.
    """
    levels = [present_levels(cfg, f) for f in factors]
    shape = tuple(len(l) for l in levels)
    assert int(np.prod(shape)) == len(cfg), f'not a complete product grid: {dict(zip(factors, shape))}'
    # The index for each factor is obtained by mapping the factor levels to their positions in the canonical order. The resulting index is a tuple of arrays, one for each factor, which can be used to index into the tensor.
    idx = tuple(cfg[f].map({l: i for i, l in enumerate(lv)}).to_numpy() for f, lv in zip(factors, levels))
    # The tensor is initialized with NaNs and then filled with the values from V at the computed indices. An assertion checks that there are no NaNs left in the tensor, which would indicate missing or duplicated configurations.
    Y = np.full(shape + (V.shape[1],), np.nan)
    Y[idx] = V
    assert not np.isnan(Y).any(), 'tensor has holes (duplicated or missing configurations)'
    return Y, levels


# --------------------------------------------------------------------------------------
# Statistics: matched-pair level comparisons
# --------------------------------------------------------------------------------------
def holm_adjust(p) -> np.ndarray:
    """Holm-Bonferroni adjustment of p-values. Returns adjusted p-values in the same order as the input."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj, running = np.empty(m), 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_pairs(Yt: np.ndarray, axis: int, alpha: float = 0.05, a_min: float = 0.56) -> pd.DataFrame:
    """
    Compute all unordered level pairs (i, j) of one factor on a tensor `Yt` (no trailing T axis).
    Matched pairs = identical values of every other factor. Reports
      mean_diff = mean(Y_i - Y_j),  A = P(Y_i > Y_j) + 0.5 P(tie)  (paired Vargha-Delaney-like effect),
      two-sided Wilcoxon signed-rank p, Holm-adjusted over the family of pairs, and
      sig = Holm p < alpha AND |A - 0.5| >= a_min - 0.5   (effect-size guard, since n is huge).
    The 20 repetitions are already collapsed to a median per configuration, and matched pairs are not
    fully independent: read p-values as screening devices and rely on A / mean_diff.
    Yt is a tensor of shape (d_1, ..., d_F) where F is the number of factors and d_j is the number of levels for each factor. The axis parameter specifies which factor to compare. It contains the values for a single n_bits.
    The function computes the mean difference and effect size (Vargha-Delaney A) for each pair of levels, performs a Wilcoxon signed-rank test, and applies Holm-Bonferroni correction to the p-values.
    The result is a DataFrame with one row per level pair, containing the computed statistics.
    """
    L = Yt.shape[axis]
    Z = np.moveaxis(Yt, axis, 0).reshape(L, -1)
    rows = []
    for i, j in itertools.combinations(range(L), 2):
        d = Z[i] - Z[j]
        nz = d[d != 0]
        p = stats.wilcoxon(nz).pvalue if nz.size >= 10 else 1.0
        rows.append(dict(i=i, j=j, mean_diff=d.mean(), A=(d > 0).mean() + 0.5 * (d == 0).mean(), p=p))
    df = pd.DataFrame(rows)
    df['p_holm'] = holm_adjust(df['p'].to_numpy())
    df['sig'] = (df['p_holm'] < alpha) & ((df['A'] - 0.5).abs() >= a_min - 0.5)
    return df


def dominance_marks(df: pd.DataFrame, L: int, higher_is_better: bool = True) -> list[str]:
    """
    Assign dominance marks to each level based on the results of paired comparisons.
    dagger: level significantly better than ALL other levels; asterisk: better than at least one. The input df is the output of paired_pairs, and L is the number of levels.
    The function computes a boolean matrix of wins for each level against all others, and then assigns a mark based on whether a level dominates all others (dagger) or at least one other (asterisk).
    If a level does not dominate any other, it receives an empty string.
    """
    wins = np.zeros((L, L), bool)
    for r in df.itertuples():
        if r.sig:
            i_better = (r.A > 0.5) == higher_is_better
            wins[(r.i, r.j) if i_better else (r.j, r.i)] = True
    marks = []
    for i in range(L):
        others = [j for j in range(L) if j != i]
        marks.append(DAG if L > 1 and wins[i, others].all() else (AST if wins[i].any() else ''))
    return marks


def normalized_regret(chosen_target: np.ndarray, target: np.ndarray) -> float:
    """0 = best configuration, 1 = average (random) configuration."""
    best, mean = target.max(), target.mean()
    return max(0.0, float((best - chosen_target) / (best - mean))) if best > mean else 0.0


def heatmap(ax, M, xlabels, ylabels, cmap='cividis', vmin=None, vmax=None, fmt='{:.2f}', fontsize=6,
            annot=None, center_white_thr=None):
    """Annotated heatmap helper. `annot` optionally overrides cell text (2D array of str)."""
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(axis='both', which='both', length=0)
    lo, hi = im.get_clim()
    cm = plt.get_cmap(cmap)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            r, g, b, _ = cm((M[i, j] - lo) / (hi - lo + 1e-12))
            col = 'white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else 'black'
            txt = annot[i][j] if annot is not None else fmt.format(M[i, j])
            ax.text(j, i, txt, ha='center', va='center', fontsize=fontsize, color=col, weight='bold')
    return im