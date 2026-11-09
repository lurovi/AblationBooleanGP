# GP hyperparameter landscape - analysis pipeline for balanced, nonlinear Boolean functions in crypto-settings

Requirements: numpy, pandas, scipy, scikit-learn, matplotlib, deap (+ LaTeX with `libertine` for `text.usetex`).
No `seaborn`, no `fanova` package is needed: fANOVA and Shapley effects are computed exactly on the complete grid.

Order of execution (all CSVs and anchors in the working dir, or set `GP_DATA_DIR`):

    python process_raw_performance_data.py
    python gp_00_prepare.py        # NotUsed + dedup -> *_main.csv (25,200) and *_dedup.csv (37,800)
    python run_all.py              # all blocks;  --only a b c d e f g  to select;  --k-perf 4 --k-stab 2 to force k
    GP_NO_TEX=1 python run_all.py  # same without LaTeX rendering (quick tests)

Outputs: `img/*.png|pdf`, `results/*.csv`, `tables/*.tex` (also printed to stdout).
Required input file for process_raw_performance_data.py: boolean_gp_full.csv
Required input files for gp_00_prepare.py: gp_bool_data_{med,std,med_scaled,std_scaled}\_pivot\_{main,dedup}.csv, upper_bounds.csv, anchors.csv, anchor_errors.csv
