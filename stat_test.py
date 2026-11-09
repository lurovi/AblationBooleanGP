import numpy as np
import statistics
import scipy.stats as stats
from statsmodels.sandbox.stats.multicomp import multipletests


def create_results_dict(scores: list[float]) -> dict[str, float]:
    res = {
        "scores": scores,
        "min": min(scores),
        "q1": float(np.percentile(scores, 25)),
        "median": statistics.median(scores),
        "q3": float(np.percentile(scores, 75)),
        "max": max(scores),
        "mean": statistics.mean(scores),
        "std": statistics.pstdev(scores)
    }
    return res


def is_kruskalwallis_passed(data: dict[str, list[float]], alpha: float = 0.05) -> bool:
    l = [data[key] for key in data]
    _, kruskal_pvalue = stats.kruskal(*l)
    return bool(kruskal_pvalue < alpha)


def is_mannwhitneyu_passed(data1: list[float], data2: list[float], alternative: str, alpha: float = 0.05) -> tuple[bool, float]:
    if data1 != data2:
        _, w_pval = stats.mannwhitneyu(data1, data2, alternative=alternative, method="auto")
        is_the_mannwhitney_test_meaningful: bool = bool(w_pval < alpha)
        return is_the_mannwhitney_test_meaningful, w_pval
    else:
        return False, 1.0


def perform_mannwhitneyu_holm_bonferroni(data: dict[str, list[float]], alternative: str, alpha: float = 0.05, method: str = 'holm') -> tuple[dict[str, bool], dict[str, dict[str, bool]]]:
    if len(data) <= 1:
        raise AttributeError(f'data must have at least two entries, found {len(data)} instead.')
    
    hb_res = {k1: False for k1 in data}
    mwu_res = {k1: {k2: False for k2 in data} for k1 in data}

    for method1 in data:
        a = data[method1]
        all_p_vals = []
        for method2 in data:
            if method2 != method1:
                b = data[method2]
                is_the_mannwhitney_test_meaningful, w_pval = is_mannwhitneyu_passed(a, b, alternative=alternative, alpha=alpha)
                all_p_vals.append(w_pval)
                mwu_res[method1][method2] = is_the_mannwhitney_test_meaningful
        all_p_vals.sort()
        reject_bonferroni, _, _, _ = multipletests(all_p_vals, alpha=alpha, method=method)
        is_the_bonferroni_test_meaningful: bool = bool(np.sum(reject_bonferroni) == len(all_p_vals))
        hb_res[method1] = is_the_bonferroni_test_meaningful
    
    if len(data) == 2:
        keys = sorted(list(data.keys()))
        first_method, second_method = keys[0], keys[1]
        hb_res[first_method] = mwu_res[first_method][second_method]
        hb_res[second_method] = mwu_res[second_method][first_method]

    return hb_res, mwu_res


def chi_square_contingency(data1: list[int], data2: list[int], alpha: float = 0.05) -> tuple[bool, float, bool]:
    for val in data1:
        if val not in (0, 1):
            raise ValueError(f'data contains non-binary values.')
    for val in data2:
        if val not in (0, 1):
            raise ValueError(f'data contains non-binary values.')

    method_A: np.ndarray = np.array(data1)
    method_B: np.ndarray = np.array(data2)

    S_A, F_A = np.sum(method_A), len(method_A) - np.sum(method_A)
    S_B, F_B = np.sum(method_B), len(method_B) - np.sum(method_B)

    p_A = S_A / len(method_A)
    p_B = S_B / len(method_B)

    table = np.array([[S_A, F_A], [S_B, F_B]])

    chi2, p, _, _ = stats.chi2_contingency(table)

    if bool(p < alpha):
        # Statistical significance detected
        if p_A > p_B:
            # Method A is better
            return True, p, True
        else:
            # Method A is NOT better (so, method B is better)
            return True, p, False
    else:
        # No statistical significance detected
        return False, p, False

def fisher_exact_test(data1: list[int], data2: list[int], alpha: float = 0.05) -> tuple[bool, float, bool]:
    for val in data1:
        if val not in (0, 1):
            raise ValueError(f'data contains non-binary values.')
    for val in data2:
        if val not in (0, 1):
            raise ValueError(f'data contains non-binary values.')

    method_A: np.ndarray = np.array(data1)
    method_B: np.ndarray = np.array(data2)

    S_A, F_A = np.sum(method_A), len(method_A) - np.sum(method_A)
    S_B, F_B = np.sum(method_B), len(method_B) - np.sum(method_B)

    p_A = S_A / len(method_A)
    p_B = S_B / len(method_B)

    table = np.array([[S_A, F_A], [S_B, F_B]])

    odds_ratio, p = stats.fisher_exact(table)

    if bool(p < alpha):
        # Statistical significance detected
        if p_A > p_B:
            # Method A is better
            return True, p, True
        else:
            # Method A is NOT better (so, method B is better)
            return True, p, False
    else:
        # No statistical significance detected
        return False, p, False
