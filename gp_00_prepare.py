import numpy as np, pandas as pd

KEY = ['TreeInit', 'PopGenRatio', 'CrossMutRate', 'Crossover', 'Mutation',
       'FunctionSet', 'TournamentSize', 'Elites']
FILES = {
    'med':        'gp_bool_data_med_pivot.csv',
    'std':        'gp_bool_data_std_pivot.csv',
    'med_scaled': 'gp_bool_data_med_scaled_pivot.csv',
    'std_scaled': 'gp_bool_data_std_scaled_pivot.csv',
}
NA_TOKEN = 'NotUsed'
MIXED = ['EqualCrossoverMutation', 'HighCrossoverLowMutation', 'LowCrossoverHighMutation']
EFF = [k for k in KEY if k != 'Crossover'] + ['Crossover_eff']

def read(path):
    df = pd.read_csv(path, dtype={k: str for k in KEY}, keep_default_na=False)
    vcols = [c for c in df.columns if c not in KEY]
    assert all(pd.api.types.is_float_dtype(df[c]) for c in vcols), "non-numeric value column"
    assert not df[vcols].isna().any().any(), "NaN in value columns"
    return df.sort_values(KEY).reset_index(drop=True)

dfs = {name: read(p) for name, p in FILES.items()}

# --- consistency across the four files
ref = dfs['med'][KEY]
for name, df in dfs.items():
    assert len(df) == 42000, f"{name}: {len(df)} rows"
    assert set(df.Elites) == {'1', '5'}
    assert df[KEY].equals(ref), f"{name}: keys differ from med file"
assert not ref.duplicated().any()

# --- structural ghost: Crossover is never called under OnlyMutation
for df in dfs.values():
    df['Crossover_eff'] = np.where(df.CrossMutRate == 'OnlyMutation', NA_TOKEN, df.Crossover)

# --- verify that the structural copies are bit-identical, in all four files
gid = dfs['med'].groupby(EFF).ngroup()
for name, df in dfs.items():
    vcols = [c for c in df.columns if c.startswith('N') and c[1:].isdigit()]
    g = df[vcols].groupby(gid)
    spread = (g.max() - g.min()).to_numpy().max()
    assert spread == 0.0, f"{name}: structural copies not identical (spread={spread})"
    print(f"{name}: max within-group spread = {spread}")

# --- drop the copies, then replace Crossover by the effective column
keep = ~dfs['med'].duplicated(subset=EFF)
assert keep.sum() == 37800, keep.sum()

out_all, out_main = {}, {}
for name, df in dfs.items():
    d = df[keep].copy()
    d['Crossover'] = d['Crossover_eff']
    d = d.drop(columns='Crossover_eff').reset_index(drop=True)
    out_all[name] = d
    m = d[d.CrossMutRate.isin(MIXED)].reset_index(drop=True)
    out_main[name] = m

# --- completeness checks
for name in FILES:
    a, m = out_all[name], out_main[name]
    assert len(a) == 37800 and len(m) == 25200
    assert not (m.Crossover == NA_TOKEN).any()
    sizes = [m[k].nunique() for k in KEY]
    assert np.prod(sizes) == len(m), f"main grid is not a complete product: {dict(zip(KEY, sizes))}"
    assert not m.duplicated(subset=KEY).any()
    a.to_csv(FILES[name].replace('.csv', '_dedup.csv'), index=False)       # 5 rates, 37,800 rows
    m.to_csv(FILES[name].replace('.csv', '_main.csv'), index=False)        # 3 mixed rates, 25,200 rows

print("all rates, deduplicated:", len(out_all['med']), "| main grid:", len(out_main['med']))