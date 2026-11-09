"""Run the whole analysis:  python run_all.py [--only a b c d e f g]   (GP_NO_TEX=1 to skip LaTeX rendering)"""
import argparse
import time

import gp_a_descriptive as A
import gp_b_importance as B
import gp_c_clustering as C
import gp_c2_cluster_robustness as C2
import gp_d_transfer as D
import gp_e_topk as E
import gp_f_boundary as F
from gp_common import load_grid

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=list('abcdefg'))
    ap.add_argument('--k-perf', type=int, default=None)
    ap.add_argument('--k-stab', type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    grid = load_grid('main')
    print(f'main grid: {grid.M} configurations, eps(log std) = {grid.eps:.3e}')
    imp = None
    steps = {
        'a': lambda: A.run(grid),
        'b': lambda: B.run(grid),
        'c': lambda: C.run(grid, args.k_perf, args.k_stab),
        'f': lambda: F.run(load_grid('dedup')),
        'g': lambda: C2.run(grid, args.k_perf, args.k_stab),
    }
    need_imp = {'d', 'e'}
    for s in 'abcdefg':
        if s not in args.only:
            continue
        print(f'\n===== block {s} =====')
        if s in need_imp:
            imp = imp or B.run(grid)
            (D.run if s == 'd' else E.run)(grid, imp)
        else:
            r = steps[s]()
            if s == 'b':
                imp = r
        print(f'[{time.time() - t0:.0f}s]')