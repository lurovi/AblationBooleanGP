import numpy as np, pandas as pd
from scipy import stats

KEY  = ['TreeInit','PopGenRatio','CrossMutRate','Crossover','Mutation',
        'FunctionSet','TournamentSize','Elites']
NS, REPS = list(range(5, 15)), 20

def _read(path):
    df = pd.read_csv(path, dtype={k: str for k in KEY})
    return df[df.Elites.isin(['1', '5'])].reset_index(drop=True)

# ---------- A. empirical ghost detector ----------
def find_ghosts(med_csv, std_csv):
    W = _read(med_csv).merge(_read(std_csv), on=KEY, suffixes=('_m', '_s'), validate='one_to_one')
    vcols = [c for c in W.columns if c not in KEY]
    W['sig'] = pd.util.hash_pandas_object(W[vcols], index=False)
    multi = W[W.groupby('sig')['sig'].transform('size') > 1]
    print(f"rows: {len(W)}, rows in duplicate groups: {len(multi)}, "
          f"groups: {multi['sig'].nunique()}")
    pattern = multi.groupby('sig')[KEY].apply(lambda d: tuple(k for k in KEY if d[k].nunique() > 1))
    print(pattern.value_counts())      # expect only ('Crossover',)
    return W

# ---------- B. paired effect of a factor inside one regime ----------
def load_long(med_csv, std_csv):
    parts = []
    for path, name in ((med_csv, 'med'), (std_csv, 'std')):
        df = _read(path)
        parts.append(df.melt(id_vars=KEY, value_vars=[f'N{n}' for n in NS],
                             var_name='N', value_name=name))
    L = parts[0].merge(parts[1], on=KEY + ['N'], validate='one_to_one')
    L['N'] = L['N'].str[1:].astype(int)
    return L

def paired_effects(L, regime, factor, ref):
    """All other factors identical (matched pairs). d is in the scaled-median unit (raw diff / 2^N)."""
    sub = L[L.CrossMutRate == regime]
    idx = [k for k in KEY if k not in ('CrossMutRate', factor)] + ['N']
    wm = sub.pivot(index=idx, columns=factor, values='med')
    ws = sub.pivot(index=idx, columns=factor, values='std')
    Ns = wm.index.get_level_values('N').to_numpy()
    rows = []
    for lvl in wm.columns:
        if lvl == ref: continue
        d  = (wm[lvl] - wm[ref]).to_numpy()
        se = 1.2533 * np.sqrt(ws[lvl].to_numpy()**2 + ws[ref].to_numpy()**2) / np.sqrt(REPS)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.where(se > 0, d / se, np.where(d == 0, 0.0, np.inf))
        ds = d / 2.0**Ns
        for n in NS:
            m = Ns == n
            nz = ds[m][ds[m] != 0]
            rows.append(dict(
                regime=regime, factor=factor, level=lvl, N=n, pairs=int(m.sum()),
                mean_d=ds[m].mean(),                    # signed: level minus reference
                mean_abs_d=np.abs(ds[m]).mean(),        # typical size of the effect
                tie=(ds[m] == 0).mean(),
                p_better=(ds[m] > 0).mean() + 0.5 * (ds[m] == 0).mean(),
                sig=(np.abs(z[m]) > 1.96).mean(),       # null expectation ~0.05
                wilcoxon_p=stats.wilcoxon(nz).pvalue if len(nz) > 10 else np.nan))
    return pd.DataFrame(rows)

def run(med_csv, std_csv, ref='MutUniform',
        thr_abs=0.005, thr_sig=0.10, thr_ratio=0.25, n_min=8):
    L = load_long(med_csv, std_csv)
    res = pd.concat([
        paired_effects(L, 'OnlyCrossover',           'Mutation', ref),
        paired_effects(L, 'EqualCrossoverMutation',  'Mutation', ref),   # calibration: mutation really active
        paired_effects(L, 'OnlyMutation',            'Mutation', ref),
        paired_effects(L, 'OnlyCrossover',           'TournamentSize', '2'),  # yardstick
    ], ignore_index=True)

    summ = (res[res.N >= n_min]
            .groupby(['factor', 'regime', 'level'])[['mean_d', 'mean_abs_d', 'sig', 'tie', 'p_better']]
            .mean().round(4))
    print(summ)

    # ratio of the effect under OnlyCrossover to the effect under Equal mix, per level and N
    mut = res[res.factor == 'Mutation']
    piv = mut.pivot_table(index=['level', 'N'], columns='regime', values='mean_abs_d')
    piv['ratio'] = piv['OnlyCrossover'] / piv['EqualCrossoverMutation']
    print(piv['ratio'].unstack('N').round(2))

    # verdict per mutation operator, on N >= n_min
    v = (mut[(mut.regime == 'OnlyCrossover') & (mut.N >= n_min)]
         .groupby('level')[['mean_abs_d', 'sig']].mean())
    v['ratio'] = piv.loc[piv.index.get_level_values('N') >= n_min, 'ratio'].groupby('level').mean()
    v['negligible'] = (v.mean_abs_d < thr_abs) & (v.sig < thr_sig) & (v.ratio < thr_ratio)
    print(v.round(4))
    return res, v

if __name__ == '__main__':
    find_ghosts('gp_bool_data_med_pivot.csv', 'gp_bool_data_std_pivot.csv')
    res, verdict = run('gp_bool_data_med_pivot.csv', 'gp_bool_data_std_pivot.csv')
    res.to_csv('mutation_effect_under_onlycrossover.csv', index=False)

    print("=====")

    W = find_ghosts('gp_bool_data_med_pivot.csv', 'gp_bool_data_std_pivot.csv')
    vcols = [c for c in W.columns if c not in KEY + ['sig']]
    g = W[W.groupby('sig')['sig'].transform('size') > 1].copy()
    pat = g.groupby('sig')[KEY].apply(lambda d: tuple(k for k in KEY if d[k].nunique() > 1))
    odd = g[g.sig.map(pat) != ('Crossover',)]
    for c in ['FunctionSet', 'PopGenRatio', 'TreeInit', 'Mutation', 'CrossMutRate', 'Elites']:
        print(odd[c].value_counts(), '\n')
    # did the run ever improve on the initial population? Proxy: std == 0 across all N in the group
    print((odd[[c for c in vcols if c.endswith('_s')]].max(axis=1) == 0).mean())

    print("=====" * 2)

    Ns  = list(range(5, 15))
    std0 = odd[[f'N{n}_s' for n in Ns]].max(axis=1) == 0
    med  = odd.loc[std0, [f'N{n}_m' for n in Ns]].to_numpy().astype(float)
    lin  = np.array([1 - 2.0**-n for n in Ns])      # balanced linear function
    and2 = np.array([-2.0**(n-2) for n in Ns])      # and/or of two variables
    tol  = 1e-3                                      # your values look float32-rounded
    print(pd.DataFrame({'linear':  np.isclose(med, lin,  atol=tol).mean(0),
                        'and2':    np.isclose(med, and2, atol=tol).mean(0)}, index=Ns))