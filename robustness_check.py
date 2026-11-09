from sklearn.metrics import adjusted_rand_score as ARI, homogeneity_score as H
from gp_common import load_grid
import gp_c_clustering as C
import gp_d_transfer as D
import pandas as pd




def cluster_robustness():
    g = load_grid('main'); X = g.score_clip
    L = {k: C.fit_ordered(X, k, X.mean(1)).labels_ for k in (2, 3, 4, 5, 6)}
    for k in (2, 3, 4, 5):
        print(f'k={k}->{k+1}: ARI={ARI(L[k], L[k+1]):.3f} | nested (homogeneity)={H(L[k], L[k+1]):.3f}')
    
    print(pd.crosstab(L[3], L[4]))

def warm_start_robustness():
    g = load_grid('main')
    print(D.warm_start(g, src=(5, 7), targets=(8, 9, 10, 11), ks=(1, 10, 100)).round(3).to_string(index=False))


cluster_robustness()
warm_start_robustness()
