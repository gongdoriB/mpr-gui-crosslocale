"""Item-bootstrap 95% CIs for REL O/R/C/F contrasts per model x language pair (dependent and independent items).
Inputs: quartet CSVs (one row per item x directed pair). 10,000 reps, items resampled within each cell, seed 42."""
import numpy as np, pandas as pd
d = pd.concat([pd.read_csv('quartets_qwen3.csv'), pd.read_csv('quartets_qwen25_internvl25.csv')])
rng = np.random.default_rng(42); REPS = 10000
C = ['R_minus_O', 'C_minus_O', 'F_minus_R', 'F_minus_C']
rows = []
for (m, dep, q, g), s in d.groupby(['model', 'text_dependency', 'query_language', 'gui_language']):
    x = s[C].to_numpy(float); n = len(x)
    idx = rng.integers(0, n, size=(REPS, n))
    boots = x[idx].mean(axis=1)  # REPS x 4
    for j, c in enumerate(C):
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
        rows.append(dict(model=m, dependency=dep, query_language=q, gui_language=g, contrast=c,
                         estimate_pp=round(100 * x[:, j].mean(), 1), ci_low_pp=round(100 * lo, 1),
                         ci_high_pp=round(100 * hi, 1), items=n, O_accuracy_pct=round(100 * s.y_O.mean(), 1)))
out = pd.DataFrame(rows); out.to_csv('pair_bootstrap.csv', index=False); print(len(out), 'rows')
