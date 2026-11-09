import numpy as np
import pandas as pd
import json

from walsh_transform import WalshTransform


DATA = pd.read_csv('boolean_gp_full.csv')
# ensure elite_size is an integer
DATA['elite_size'] = DATA['elite_size'].astype(int)
# exclude all rows with elite_size == 10
DATA = DATA[DATA['elite_size'] != 10]
print(DATA.head(50))
print(DATA.columns)



"""
#groupt by n_bits and get the maximum best_fitness for each n_bits
max_fitness_per_n_bits = DATA.groupby('n_bits')['best_fitness'].max()
print(max_fitness_per_n_bits)

Maximum fitness values for each number of bits in the boolean GP problem, across all configurations and repetitions, according to the content of DATA.
All values reach the maximum possible known fitness, except for n_bits=12 and n_bits=13, where the maximum are respectively 2012 and 4036.

        best_fitness
n_bits              
5          12.625000
6          26.750000
7          56.734375
8         116.937500
9         240.734375
10        492.812500
11        992.718750
12       2010.875000
13       4032.687500
14       8120.750000
"""

"""
Function sets:

all -> and_or_xor_not_nand_nor_xnor_and2_if
custom1 -> or_xor_and_and2_xnor_if_not
minimal -> nand
basic_with_if -> and_or_not_if
basic -> and_or_not
"""

"""
mutUniform:0:2 -> mutUniform
selTournament:4 -> 4
selTournament:2 -> 2, etc.
"""


DATA = DATA[['n_bits', 'population_size', 'num_generations', 'crossover_prob', 'mutation_prob', 'elite_size', 'tree_init_method', 'selection_operator', 'crossover_operator', 'mutation_operator', 'operators_key', 'operators', 'best_fitness', 'best_tree_size']]
 
# print unique crossover_prob mutation_prob combinations
print(DATA[['crossover_prob', 'mutation_prob']].drop_duplicates())

def map_cross_mut_prob_to_category(crossover_prob, mutation_prob):
    if crossover_prob == 1.0 and mutation_prob == 0.0:
        return 'onlyCrossover'
    elif crossover_prob == 0.0 and mutation_prob == 1.0:
        return 'onlyMutation'
    elif crossover_prob == 0.5 and mutation_prob == 0.5:
        return 'equalCrossoverMutation'
    elif crossover_prob == 0.7 and mutation_prob == 0.3:
        return 'highCrossoverLowMutation'
    elif crossover_prob == 0.3 and mutation_prob == 0.7:
        return 'lowCrossoverHighMutation'
    else:
        raise ValueError(f"Unknown crossover_prob and mutation_prob combination: {crossover_prob}, {mutation_prob}")

# create new attribute 'cross_mut_category' in DATA
DATA['cross_mut_category'] = DATA.apply(lambda row: map_cross_mut_prob_to_category(row['crossover_prob'], row['mutation_prob']), axis=1)
# drop the original crossover_prob and mutation_prob columns
DATA = DATA.drop(columns=['crossover_prob', 'mutation_prob'])

# print unique population_size num_generations combinations
print(DATA[['population_size', 'num_generations']].drop_duplicates())

def map_pop_gen_to_category(population_size, num_generations):
    if num_generations == 0:
        raise ValueError("num_generations cannot be zero to avoid division by zero.")
    return f"{population_size / num_generations}"

# create new attribute 'pop_gen_ratio' in DATA
DATA['pop_gen_ratio'] = DATA.apply(lambda row: map_pop_gen_to_category(row['population_size'], row['num_generations']), axis=1)
# drop the original population_size and num_generations columns
DATA = DATA.drop(columns=['population_size', 'num_generations'])

print(DATA['selection_operator'].unique())
print(DATA['tree_init_method'].unique())
print(DATA['crossover_operator'].unique())
print(DATA['mutation_operator'].unique())
print(DATA['operators_key'].unique())

# considering that selection_operator is selTournament:4 with only the last digit changing, we can create a new attribute 'tournament_size' in DATA that only keeps the number
DATA['tournament_size'] = DATA['selection_operator'].str.split(':').str[-1]
print(DATA['tournament_size'].unique())
# drop the original selection_operator column
DATA = DATA.drop(columns=['selection_operator'])

print(DATA.head(50))
print(DATA.columns)

# in the mutation_operator column, replace all occurrences of 'mutUniform:0:2' with 'mutUniform'
DATA['mutation_operator'] = DATA['mutation_operator'].replace('mutUniform:0:2', 'mutUniform')
print(DATA['mutation_operator'].unique())

# Drop operator columns that are not needed for analysis
DATA = DATA.drop(columns=['operators'])
# Rename operators_key to function_set for clarity
DATA = DATA.rename(columns={'operators_key': 'function_set'})
# replace function_set values to be more descriptive
DATA['function_set'] = DATA['function_set'].replace({
    'all': 'all',
    'custom1': 'custom',
    'minimal': 'minimal',
    'basic_with_if': 'basicWithIf',
    'basic': 'basic'
})
print(DATA['function_set'].unique())

# drop best_tree_size column as it is not needed for analysis
DATA = DATA.drop(columns=['best_tree_size'])

# convert all values except for best_fitness and n_bits to string type
for col in DATA.columns:
    if col not in ['best_fitness', 'n_bits']:
        DATA[col] = DATA[col].astype(str)
        # capitalize the first letter of each word in the column values
        DATA[col] = DATA[col].str[0].str.upper() + DATA[col].str[1:]

# change column names to custom names
DATA = DATA.rename(columns={
    'n_bits': 'N',
    'elite_size': 'Elites',
    'tree_init_method': 'TreeInit',
    'crossover_operator': 'Crossover',
    'mutation_operator': 'Mutation',
    'function_set': 'FunctionSet',
    'cross_mut_category': 'CrossMutRate',
    'pop_gen_ratio': 'PopGenRatio',
    'best_fitness': 'BestFitness',
    'tournament_size': 'TournamentSize'
})

print(DATA.head(50))
DATA = DATA[['N', 'TreeInit', 'PopGenRatio', 'CrossMutRate', 'Crossover', 'Mutation', 'FunctionSet', 'TournamentSize', 'Elites', 'BestFitness']]
print(DATA.head(50))

DATA_MED = DATA.copy()
DATA_STD = DATA.copy()

print(DATA_MED.shape)
print(DATA_STD.shape)

# group by N, TreeInit, PopGenRatio, CrossMutRate, Crossover, Mutation, FunctionSet, TournamentSize, Elites and calculate the median and std of BestFitness (median for DATA_MED and std for DATA_STD)
DATA_MED = DATA_MED.groupby(['N', 'TreeInit', 'PopGenRatio', 'CrossMutRate', 'Crossover', 'Mutation', 'FunctionSet', 'TournamentSize', 'Elites'], as_index=False).median()
DATA_STD = DATA_STD.groupby(['N', 'TreeInit', 'PopGenRatio', 'CrossMutRate', 'Crossover', 'Mutation', 'FunctionSet', 'TournamentSize', 'Elites'], as_index=False).std()

print(DATA_MED.head(50))
print(DATA_STD.head(50))

print(DATA_MED.shape)
print(DATA_STD.shape)

cnt = DATA.groupby(['N','TreeInit','PopGenRatio','CrossMutRate','Crossover',
                    'Mutation','FunctionSet','TournamentSize','Elites']).size()
assert len(cnt) == 10 * 42000 and (cnt == 20).all()
assert DATA_STD['BestFitness'].notna().all()

# print distinct values for each column in DATA_MED
for column in DATA_MED.columns:
    print(f"Column {column}: {DATA_MED[column].unique()}")

for column in DATA_STD.columns:
    print(f"Column {column}: {DATA_STD[column].unique()}")

U = {5:12.625, 6:26.75, 7:56.734375, 8:116.9375, 9:240.734375,
     10:492.8125, 11:992.71875, 12:2012.0, 13:4036.0, 14:8120.75}

def anchors(n, reps, seed=0):
    wt, N = WalshTransform(n), 2**n
    rng = np.random.default_rng(seed)
    f = np.empty(reps)
    for r in range(reps):
        tt = np.zeros(N, dtype=np.int32); tt[rng.choice(N, N//2, replace=False)] = 1
        spec, _ = wt.apply(tt)
        f[r] = wt.granular_non_linearity_with_balancedness_penalty(tt, spec)
    return f.mean(), f.std(ddof=1) / np.sqrt(reps)

REPS = {n: (20000 if n <= 10 else 20000) for n in range(5, 15)}
AN = {n: anchors(n, REPS[n]) for n in range(5, 15)}
B  = {n: a for n, (a, _) in AN.items()}
B_err = {n: e for n, (_, e) in AN.items()}

# save U, B, and B_err to csv files
pd.DataFrame.from_dict(U, orient='index', columns=['U']).to_csv('upper_bounds.csv')
pd.DataFrame.from_dict(B, orient='index', columns=['B']).to_csv('anchors.csv')
pd.DataFrame.from_dict(B_err, orient='index', columns=['B_err']).to_csv('anchor_errors.csv')

json.dump({'seed': 0, 'reps': 20000, 'estimator': 'mean', 'ddof': 1}, open('anchors_meta.json', 'w'))

scale_med = lambda f, n: (f - B[n]) / (U[n] - B[n])
scale_std = lambda s, n: s / (U[n] - B[n])         # slope only

DATA_MED_SCALED = DATA_MED.copy()
DATA_STD_SCALED = DATA_STD.copy()

# ensure N is integer and BestFitness is float
DATA_MED_SCALED['N'] = DATA_MED_SCALED['N'].astype(int)
DATA_STD_SCALED['N'] = DATA_STD_SCALED['N'].astype(int)
DATA_MED_SCALED['BestFitness'] = DATA_MED_SCALED['BestFitness'].astype(float)
DATA_STD_SCALED['BestFitness'] = DATA_STD_SCALED['BestFitness'].astype(float)

# apply scale_med to DATA_MED_SCALED['BestFitness'] and scale_std to DATA_STD_SCALED['BestFitness']
DATA_MED_SCALED['BestFitness'] = DATA_MED_SCALED.apply(lambda row: scale_med(row['BestFitness'], row['N']), axis=1)
DATA_STD_SCALED['BestFitness'] = DATA_STD_SCALED.apply(lambda row: scale_std(row['BestFitness'], row['N']), axis=1)

DATA_MED_PIVOT = DATA_MED.pivot(index=['TreeInit', 'PopGenRatio', 'CrossMutRate', 'Crossover', 'Mutation', 'FunctionSet', 'TournamentSize', 'Elites'], columns='N', values='BestFitness')
DATA_STD_PIVOT = DATA_STD.pivot(index=['TreeInit', 'PopGenRatio', 'CrossMutRate', 'Crossover', 'Mutation', 'FunctionSet', 'TournamentSize', 'Elites'], columns='N', values='BestFitness')

DATA_MED_SCALED_PIVOT = DATA_MED_SCALED.pivot(index=['TreeInit', 'PopGenRatio', 'CrossMutRate', 'Crossover', 'Mutation', 'FunctionSet', 'TournamentSize', 'Elites'], columns='N', values='BestFitness')
DATA_STD_SCALED_PIVOT = DATA_STD_SCALED.pivot(index=['TreeInit', 'PopGenRatio', 'CrossMutRate', 'Crossover', 'Mutation', 'FunctionSet', 'TournamentSize', 'Elites'], columns='N', values='BestFitness')

# flatten index of DATA_MED_PIVOT and DATA_STD_PIVOT
DATA_MED_PIVOT = DATA_MED_PIVOT.reset_index()
DATA_STD_PIVOT = DATA_STD_PIVOT.reset_index()

DATA_MED_SCALED_PIVOT = DATA_MED_SCALED_PIVOT.reset_index()
DATA_STD_SCALED_PIVOT = DATA_STD_SCALED_PIVOT.reset_index()

DATA_MED_PIVOT = DATA_MED_PIVOT.rename(columns={5: 'N5', 6: 'N6', 7: 'N7', 8: 'N8', 9: 'N9', 10: 'N10', 11: 'N11', 12: 'N12', 13: 'N13', 14: 'N14'})
DATA_STD_PIVOT = DATA_STD_PIVOT.rename(columns={5: 'N5', 6: 'N6', 7: 'N7', 8: 'N8', 9: 'N9', 10: 'N10', 11: 'N11', 12: 'N12', 13: 'N13', 14: 'N14'})

DATA_MED_SCALED_PIVOT = DATA_MED_SCALED_PIVOT.rename(columns={5: 'N5', 6: 'N6', 7: 'N7', 8: 'N8', 9: 'N9', 10: 'N10', 11: 'N11', 12: 'N12', 13: 'N13', 14: 'N14'})
DATA_STD_SCALED_PIVOT = DATA_STD_SCALED_PIVOT.rename(columns={5: 'N5', 6: 'N6', 7: 'N7', 8: 'N8', 9: 'N9', 10: 'N10', 11: 'N11', 12: 'N12', 13: 'N13', 14: 'N14'})

# REMOVE THE COLUMN NAME 'N' FROM THE INDEX COLUMN (0, 1, 2, 3, 4, 5, 6, 7, 8, ...) OF DATA_MED_PIVOT AND DATA_STD_PIVOT

DATA_MED_PIVOT.columns.name = None
DATA_STD_PIVOT.columns.name = None

DATA_MED_SCALED_PIVOT.columns.name = None
DATA_STD_SCALED_PIVOT.columns.name = None

print(DATA_MED_PIVOT.head(50))
print(DATA_STD_PIVOT.head(50))

print(DATA_MED_SCALED_PIVOT.head(50))
print(DATA_STD_SCALED_PIVOT.head(50))

# print unique values for each column in DATA_MED_PIVOT
for column in DATA_MED_PIVOT.columns:
    print(f"Column {column}: {DATA_MED_PIVOT[column].unique()}")

# persist all of the 4 dataframes to csv files
DATA_MED_PIVOT.to_csv('gp_bool_data_med_pivot.csv', index=False)
DATA_STD_PIVOT.to_csv('gp_bool_data_std_pivot.csv', index=False)
DATA_MED_SCALED_PIVOT.to_csv('gp_bool_data_med_scaled_pivot.csv', index=False)
DATA_STD_SCALED_PIVOT.to_csv('gp_bool_data_std_scaled_pivot.csv', index=False)



U = pd.read_csv('upper_bounds.csv', index_col=0)['U']
B = pd.read_csv('anchors.csv', index_col=0)['B']
E = pd.read_csv('anchor_errors.csv', index_col=0)['B_err']
for n in U.index:
    g = U[n] - B[n]; s = (1 - 2.0**-n - B[n]) / g
    dag = r'$^{\dagger}$' if n == 13 else ''
    print(f"{n} & {U[n]:.4f}{dag} & {B[n]:.4f} & {g:.4f} & {E[n]:.4f} "
          f"& {100*E[n]/g:.2f} & {100*2/g:.1f} & ${s:.2f}$ \\\\")
