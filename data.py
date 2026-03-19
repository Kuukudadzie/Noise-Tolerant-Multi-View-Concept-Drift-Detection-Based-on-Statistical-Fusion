"""
data.py
=======
Synthetic data stream generators for concept drift benchmarking.

Datasets
--------
SEA        — threshold-based binary classification (3 features)
SINE       — sinusoidal boundary classification (4 features, 2 irrelevant)
Hyperplane — rotating hyperplane classification (10 features)
AGRAWAL    — loan function switching (9 features)

All generators share the same interface:

    X_all, y_all = generate_<name>(
        n_chunks   : int   = number of stream chunks
        chunk_size : int   = samples per chunk
        drift_chunks : list = chunk indices where concept switches
        noise      : float = symmetric label noise rate η ∈ [0, 1)
    )

Returns
-------
X_all : list of np.ndarray, shape (chunk_size, n_features)
y_all : list of np.ndarray, shape (chunk_size,), dtype int

Label noise
-----------
With rate η, each label is independently flipped to a uniformly
sampled incorrect class.  Applied after concept generation so that
the ground-truth concept boundaries are noise-free.

Usage
-----
    from data import generate_sea, generate_sine
    from data import generate_hyperplane, generate_agrawal

    X_all, y_all = generate_sea(
        n_chunks=60, chunk_size=200,
        drift_chunks=[12, 24, 36, 48], noise=0.10
    )
"""

import numpy as np


# ── SEA ───────────────────────────────────────────────────────────────────────

def generate_sea(n_chunks, chunk_size, drift_chunks, noise=0.0):
    """
    SEA dataset (Street & Kim, 2001).

    Three features x1, x2, x3 ∈ [0, 10].
    Label: y = 1  if  x1 + x2 <= threshold[concept]  else  0.
    Concepts rotate through four thresholds: [8, 9, 7, 9.5].

    Parameters
    ----------
    n_chunks     : total number of chunks
    chunk_size   : samples per chunk
    drift_chunks : chunk indices at which the concept switches
    noise        : symmetric label noise rate η
    """
    thresholds = [8, 9, 7, 9.5]
    concept = 0
    X_all, y_all = [], []
    for ci in range(n_chunks):
        if ci in drift_chunks:
            concept = (concept + 1) % len(thresholds)
        X = np.random.uniform(0, 10, (chunk_size, 3))
        y = (X[:, 0] + X[:, 1] <= thresholds[concept]).astype(int)
        if noise > 0:
            flip = np.random.random(chunk_size) < noise
            y[flip] = 1 - y[flip]
        X_all.append(X)
        y_all.append(y)
    return X_all, y_all


# ── SINE ──────────────────────────────────────────────────────────────────────

def generate_sine(n_chunks, chunk_size, drift_chunks, noise=0.0):
    """
    SINE dataset.

    Four features: x1, x2 ∈ [0, 1] (relevant) + two irrelevant uniform.
    Concept 0: y = 1  if  x2 < sin(π·x1)
    Concept 1: y = 1  if  x2 >= sin(π·x1)
    Concepts alternate at each drift point.

    Parameters
    ----------
    n_chunks     : total number of chunks
    chunk_size   : samples per chunk
    drift_chunks : chunk indices at which the concept switches
    noise        : symmetric label noise rate η
    """
    concept = 0
    X_all, y_all = [], []
    for ci in range(n_chunks):
        if ci in drift_chunks:
            concept = 1 - concept
        x1 = np.random.uniform(0, 1, chunk_size)
        x2 = np.random.uniform(0, 1, chunk_size)
        irr1 = np.random.uniform(0, 1, chunk_size)
        irr2 = np.random.uniform(0, 1, chunk_size)
        X = np.c_[x1, x2, irr1, irr2]
        boundary = np.sin(x1 * np.pi)
        y = ((x2 < boundary) == (concept == 0)).astype(int)
        if noise > 0:
            flip = np.random.random(chunk_size) < noise
            y[flip] = 1 - y[flip]
        X_all.append(X)
        y_all.append(y)
    return X_all, y_all


# ── HYPERPLANE ────────────────────────────────────────────────────────────────

def generate_hyperplane(n_chunks, chunk_size, drift_chunks,
                        noise=0.0, n_features=10):
    """
    Rotating Hyperplane dataset.

    n_features features in [-1, 1].
    Label: y = 1  if  X @ weights > 0  else  0.
    At each drift point the weight vector is negated (180° rotation),
    producing an abrupt concept change that flips all labels.

    Parameters
    ----------
    n_chunks     : total number of chunks
    chunk_size   : samples per chunk
    drift_chunks : chunk indices at which the hyperplane rotates
    noise        : symmetric label noise rate η
    n_features   : dimensionality of the feature space (default 10)
    """
    weights = np.ones(n_features)
    X_all, y_all = [], []
    for ci in range(n_chunks):
        if ci in drift_chunks:
            weights = -weights
        X = np.random.uniform(-1, 1, (chunk_size, n_features))
        y = (X @ weights > 0).astype(int)
        if noise > 0:
            flip = np.random.random(chunk_size) < noise
            y[flip] = 1 - y[flip]
        X_all.append(X)
        y_all.append(y)
    return X_all, y_all


# ── AGRAWAL ───────────────────────────────────────────────────────────────────

def generate_agrawal(n_chunks, chunk_size, drift_chunks,
                     noise=0.0, functions=None):
    """
    AGRAWAL loan dataset (Agrawal et al., 1993).

    Nine features: salary, commission, age, elevel, car, zipcode,
    hvalue, hyears, loan.
    Ten classification functions f1–f10 represent different loan
    approval criteria.  At each drift point the active function
    advances cyclically through the provided list.

    Parameters
    ----------
    n_chunks     : total number of chunks
    chunk_size   : samples per chunk
    drift_chunks : chunk indices at which the function switches
    noise        : symmetric label noise rate η
    functions    : list of function IDs to cycle through
                   (default [1, 3, 5, 7, 9])
    """
    if functions is None:
        functions = [1, 3, 5, 7, 9]

    def _apply(fid, sal, age, hval):
        if   fid == 1:  return int((age < 40) and (sal >= 50000))
        elif fid == 2:  return int((age < 40) and (50000 <= sal < 100000))
        elif fid == 3:  return int((age < 40) or (sal < 50000))
        elif fid == 4:  return int(((40 <= age <= 60)) or (sal >= 100000))
        elif fid == 5:  return int((40 <= age <= 60) and (sal >= 100000))
        elif fid == 6:  return int((age < 40) or (50000 <= sal < 100000))
        elif fid == 7:  return int(((age < 40) and (sal < 50000)) or
                                   ((age >= 60) and (sal >= 100000)))
        elif fid == 8:  return int(((age < 40) or (sal >= 100000)) and
                                   (hval > 200000))
        elif fid == 9:  return int(((age >= 60) and (sal < 50000)) or
                                   (hval < 150000))
        elif fid == 10: return int((age >= 40) and (sal >= 50000))
        return 0

    concept_idx = 0
    X_all, y_all = [], []
    for ci in range(n_chunks):
        if ci in drift_chunks:
            concept_idx = (concept_idx + 1) % len(functions)
        fid = functions[concept_idx]
        rows, labels = [], []
        for _ in range(chunk_size):
            sal    = np.random.uniform(20000, 150000)
            age    = np.random.uniform(20, 80)
            zc     = np.random.randint(0, 9)
            hv     = np.random.uniform(50000 * (zc + 1), 100000 * (zc + 1))
            rows.append([
                sal,
                0 if sal >= 75000 else np.random.uniform(10000, 75000),
                age,
                np.random.randint(0, 5),
                np.random.randint(1, 21),
                zc, hv,
                np.random.uniform(1, 30),
                np.random.uniform(0, 500000),
            ])
            labels.append(_apply(fid, sal, age, hv))
        X = np.array(rows)
        y = np.array(labels)
        if noise > 0:
            flip = np.random.random(chunk_size) < noise
            y[flip] = 1 - y[flip]
        X_all.append(X)
        y_all.append(y)
    return X_all, y_all


# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "SEA": generate_sea,
    "SINE": generate_sine,
    "HYPERPLANE": lambda nc, cs, dc, n: generate_hyperplane(nc, cs, dc, n, 10),
    "AGRAWAL": lambda nc, cs, dc, n: generate_agrawal(nc, cs, dc, n),
}
"""
Dict mapping dataset name → generator callable with signature
(n_chunks, chunk_size, drift_chunks, noise).
"""
