# MV-Fuse: Noise-Tolerant Multi-View Concept Drift Detection

**Paper:** *Noise-Tolerant Multi-View Concept Drift Detection Based on Statistical Fusion*  
**Authors:** Benjamin Mensah Dadzie · Piotr Porwik  
**Affiliation:** Faculty of Science and Technology, University of Silesia in Katowice, Poland  
**Venue:** Procedia Computer Science, 2025

---

## Overview

**MV-Fuse** is a two-stage concept drift detector designed to distinguish genuine distributional shifts from noise-driven artefacts in supervised data streams.

### How it works

| Stage | What it does |
|-------|-------------|
| **Stage I** | Prequential accuracy monitoring flags candidate change points when the lower confidence bound of current accuracy falls below the historical best |
| **Stage II** | Three statistical views are computed over adjacent windows W⁻/W⁺ and their p-values are fused via Brown's method |

The three views are:

- **View 1 — Decision margins:** KS test on distributions of `p₍₁₎(x) − p₍₂₎(x)` (top-two posterior gap) → p₁  
- **View 2 — Predictive entropy:** KS test on Shannon entropy distributions → p₂  
- **View 3 — Label frequency:** χ² test on label count vectors → p₃  

Brown's method combines these correlated p-values accounting for the covariance between Views 1 and 2 (both derived from the same classifier posterior). Drift is confirmed only when all three views agree.

---

## Results (overall — 4 datasets × 5 noise levels × 5 reps = 100 conditions)

| Method | TP | FP | FN | TPR | Precision | **F1** | EDDR |
|--------|----|----|-----|-----|-----------|--------|------|
| **MV-Fuse** | **3.5** | **1.0** | **0.5** | 0.87 | **0.77** | **0.799** | **0.709** |
| DDM | 2.6 | 1.7 | 1.4 | 0.66 | 0.58 | 0.592 | 0.483 |
| HDDM-W | 1.6 | 0.1 | 2.5 | 0.39 | 0.42 | 0.398 | 0.377 |
| EDDM | 3.4 | 12.5 | 0.6 | 0.85 | 0.24 | 0.357 | 0.229 |
| HDDM-A | 3.4 | 16.9 | 0.6 | 0.86 | 0.19 | 0.300 | 0.182 |
| ADWIN | 0.0 | 0.0 | 4.0 | 0.00 | 0.00 | 0.000 | 0.000 |

All improvements over baselines are statistically significant (p < 0.001, Wilcoxon signed-rank test).

---

## Repository structure

```
mv-fuse/
├── data.py              # Dataset generators (SEA, SINE, Hyperplane, AGRAWAL)
├── run_experiment.py    # Full experiment runner → saves results.pkl + results.json
├── plot_results.py      # Generates 6 publication figures from results.pkl
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the experiment

```bash
python run_experiment.py
```

This prints per-dataset, per-noise-level tables to the terminal and saves:
- `results.pkl` — full per-replication data (used by `plot_results.py`)
- `results.json` — human-readable averaged summaries

**Expected runtime:** ~8–12 minutes on a modern CPU (4 datasets × 5 noise levels × 5 reps).

### 3. Generate figures

```bash
python plot_results.py
```

Reads `results.pkl` and saves six PNG figures. To use a different results file:

```bash
python plot_results.py path/to/my_results.pkl
```

---

## Reproducibility

All results are fully reproducible. Each replication `r` uses random seed `r * 100 + 7`, applied before both data generation and classifier initialisation. Running `run_experiment.py` twice on the same machine (and the same library versions) produces identical numbers.

Tested with:
- Python 3.10+
- NumPy 2.4.2
- SciPy 1.17.0
- scikit-learn 1.8.0
- matplotlib 3.x

---

## Datasets

All four datasets are generated synthetically using the functions in `data.py`. No external data files are required.

| Dataset | Features | Concept change | Reference |
|---------|----------|----------------|-----------|
| SEA | 3 | Threshold rotation | Street & Kim (2001) |
| SINE | 4 (2 relevant) | Sinusoidal boundary flip | — |
| Hyperplane | 10 | Weight vector negation | — |
| AGRAWAL | 9 | Loan function switching | Agrawal et al. (1993) |

Each stream: 60 chunks × 200 samples, drift points at chunks {12, 24, 36, 48}.  
Noise: symmetric label noise at η ∈ {0%, 5%, 10%, 15%, 20%}.

---

## Algorithm parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `window_size` | w | 5 | Chunks in each comparison window W⁻/W⁺ |
| `alpha` | α | 0.05 | Brown's combined p-value threshold |
| `tau_p` | τₚ | 1 | Consecutive confirmations required |
| `tau_c` | τ꜀ | 5 | Cooldown chunks after confirmed drift |
| `lam` | λ | 1.0 | Stage I sensitivity multiplier |
| `ref_size` | n₀ | 50 | Min. observations for covariance estimation |

---

## Using MV-Fuse in your own code

```python
from run_experiment import MVFuse, calc_metrics
from data import generate_sea
from sklearn.naive_bayes import GaussianNB
import numpy as np

np.random.seed(42)
X_all, y_all = generate_sea(n_chunks=60, chunk_size=200,
                             drift_chunks=[12, 24, 36, 48], noise=0.1)
classes = np.unique(np.concatenate(y_all))

mv  = MVFuse(window_size=5, alpha=0.05, tau_p=1, tau_c=5, lam=1.0)
clf = GaussianNB()
clf.partial_fit(X_all[0], y_all[0], classes=classes)

detections = []
for i in range(1, 60):
    X, y   = X_all[i], y_all[i]
    pred   = clf.predict(X)
    proba  = clf.predict_proba(X)
    mv.feed(X, y, pred, proba)
    if mv.drift[-1] == 2:
        detections.append(i)
        clf = GaussianNB()
        clf.partial_fit(X, y, classes=classes)
    else:
        clf.partial_fit(X, y)

metrics = calc_metrics([12, 24, 36, 48], detections, n_chunks=60)
print(f"Detections: {detections}")
print(f"F1={metrics['F1']:.3f}  TP={metrics['TP']}  FP={metrics['FP']}")
```

---

## Citation

If you use this code or results, please cite:

```bibtex
@article{dadzie2025mvfuse,
  title   = {Noise-Tolerant Multi-View Concept Drift Detection
             Based on Statistical Fusion},
  author  = {Dadzie, Benjamin Mensah and Porwik, Piotr},
  journal = {Procedia Computer Science},
  year    = {2025}
}
```

---

## License

MIT License. See `LICENSE` for details.
