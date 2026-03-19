"""
run_experiment.py
=================
Reproduces the full MV-Fuse benchmark from:

  "Noise-Tolerant Multi-View Concept Drift Detection
   Based on Statistical Fusion"
  B. M. Dadzie & P. Porwik — Procedia Computer Science, 2025

Runs 4 datasets × 5 noise levels × 5 replications = 100 conditions
and saves results to:

    results.pkl   — full per-rep data (loaded by plot_results.py)
    results.json  — human-readable averaged summaries

Usage
-----
    python run_experiment.py

Requirements
------------
    numpy >= 1.21
    scipy >= 1.7
    scikit-learn >= 1.0
"""

import json
import pickle
import warnings

import numpy as np
from scipy.stats import chi2, chi2_contingency, ks_2samp, wilcoxon
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB

from data import DATASETS

warnings.filterwarnings("ignore")


# ── Random seed used per replication ─────────────────────────────────────────
# Replication r uses seed  r * 100 + 7  (r = 0 … n_reps-1)
# This ensures full reproducibility across machines and library versions.


# ==============================================================================
# Brown's method — combine correlated p-values
# ==============================================================================

def browns_method(pvalues, cov_matrix=None):
    """
    Combine k p-values using Brown's method (Brown 1975).

    Extends Fisher's statistic X² = -2 Σ ln(pᵢ) to correlated tests
    by scaling it with effective degrees of freedom estimated from the
    empirical covariance of the log-transformed p-values.

    Falls back to standard Fisher fusion when cov_matrix is None or
    the reference window contains fewer than ref_size observations.

    Parameters
    ----------
    pvalues    : array-like of k p-values in (0, 1]
    cov_matrix : (k, k) empirical covariance of (-2 ln pᵢ) values,
                 or None for Fisher fallback

    Returns
    -------
    p_combined : float, combined p-value
    """
    pv  = np.clip(np.asarray(pvalues, dtype=float), 1e-15, 1.0)
    k   = len(pv)
    t_i = -2.0 * np.log(pv)
    X2  = float(np.sum(t_i))

    if cov_matrix is not None and np.asarray(cov_matrix).shape == (k, k):
        cov = np.asarray(cov_matrix, dtype=float)
        E   = 2.0 * k
        V   = np.sum(np.diag(cov)) + 2.0 * np.sum(np.triu(cov, k=1))
        if V > 0 and E > 0:
            c = V / (2.0 * E)
            f = (2.0 * E ** 2) / V
            return float(np.clip(1.0 - chi2.cdf(X2 / c, df=f), 0.0, 1.0))

    # Fisher fallback (independence assumption)
    return float(np.clip(1.0 - chi2.cdf(X2, df=2 * k), 0.0, 1.0))


# ==============================================================================
# MV-Fuse detector
# ==============================================================================

class MVFuse(BaseEstimator):
    """
    Multi-View Fusion drift detector.

    Stage I  — prequential accuracy monitoring flags candidate points.
    Stage II — three-view statistical validation:
                 View 1: KS test on decision margin distributions → p1
                 View 2: KS test on predictive entropy distributions → p2
                 View 3: χ² test on label frequency counts          → p3
               p-values are fused via Brown's method.

    Drift is confirmed when p_fuse < α for τ_p consecutive candidates,
    followed by a cooldown of τ_c chunks.

    Optimal parameters (determined by grid search):
        window_size=5, alpha=0.05, tau_p=1, tau_c=5, lam=1.0

    Parameters
    ----------
    window_size : w   — chunks in each comparison window W⁻ / W⁺
    alpha       :     — Brown's combined p-value threshold
    tau_p       :     — consecutive confirmations required
    tau_c       :     — cooldown chunks after a confirmed drift
    lam         : λ   — Stage I sensitivity multiplier
    ref_size    : n₀  — minimum triples to estimate covariance
                        (falls back to Fisher when fewer available)
    """

    def __init__(self, window_size=5, alpha=0.05, tau_p=1,
                 tau_c=5, lam=1.0, ref_size=50):
        self.window_size = window_size
        self.alpha       = alpha
        self.tau_p       = tau_p
        self.tau_c       = tau_c
        self.lam         = lam
        self.ref_size    = ref_size
        self._reset()

    # ------------------------------------------------------------------
    def _reset(self):
        self.chunk_idx   = 0
        self.cum_correct = 0
        self.cum_total   = 0
        self.p_max       = 0.0
        self.s_max       = 0.0
        self.margin_buf  = []   # per-chunk margin arrays  (View 1)
        self.entropy_buf = []   # per-chunk entropy arrays (View 2)
        self.label_buf   = []   # per-chunk label arrays   (View 3)
        self.ref_log_p   = []   # (-2 ln p) triples for covariance
        self.cooldown    = 0
        self.consecutive = 0
        self.drift       = []   # 0 = no drift, 2 = drift per chunk

    # ------------------------------------------------------------------
    @staticmethod
    def _margins(proba):
        """p_(1) − p_(2) per sample."""
        s = np.sort(proba, axis=1)
        return s[:, -1] - s[:, -2] if proba.shape[1] >= 2 else np.zeros(len(proba))

    @staticmethod
    def _entropy(proba):
        """Shannon entropy per sample."""
        p = np.clip(proba, 1e-15, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    # ------------------------------------------------------------------
    def _estimate_cov(self):
        """3×3 empirical covariance of (-2 ln p) triples, or None."""
        if len(self.ref_log_p) < self.ref_size:
            return None
        return np.cov(np.array(self.ref_log_p), rowvar=False)

    # ------------------------------------------------------------------
    def _validate(self, Wm, Wp):
        """Run three-view tests and return Brown's combined p-value."""
        # View 1 — KS on decision margins
        m_neg = np.concatenate(Wm["margins"])
        m_pos = np.concatenate(Wp["margins"])
        p1 = ks_2samp(m_neg, m_pos).pvalue if (len(m_neg) > 1 and len(m_pos) > 1) else 1.0

        # View 2 — KS on predictive entropy
        h_neg = np.concatenate(Wm["entropy"])
        h_pos = np.concatenate(Wp["entropy"])
        p2 = ks_2samp(h_neg, h_pos).pvalue if (len(h_neg) > 1 and len(h_pos) > 1) else 1.0

        # View 3 — χ² on label frequency counts
        labs_neg = np.concatenate(Wm["labels"])
        labs_pos = np.concatenate(Wp["labels"])
        classes  = np.unique(np.concatenate([labs_neg, labs_pos]))
        if len(classes) < 2:
            p3 = 1.0
        else:
            n_neg = np.array([np.sum(labs_neg == c) for c in classes])
            n_pos = np.array([np.sum(labs_pos == c) for c in classes])
            ct    = np.vstack([n_neg, n_pos])
            nz    = ct.sum(axis=0) > 0
            try:
                p3 = chi2_contingency(ct[:, nz]).pvalue if nz.sum() >= 2 else 1.0
            except Exception:
                p3 = 1.0

        # Accumulate log-p triple for covariance estimation
        t = np.array([
            -2.0 * np.log(max(p1, 1e-15)),
            -2.0 * np.log(max(p2, 1e-15)),
            -2.0 * np.log(max(p3, 1e-15)),
        ])
        self.ref_log_p.append(t)

        return browns_method([p1, p2, p3], cov_matrix=self._estimate_cov())

    # ------------------------------------------------------------------
    def feed(self, X, y, pred, proba):
        """
        Process one chunk.

        Parameters
        ----------
        X     : (n, d) feature matrix
        y     : (n,)   true labels
        pred  : (n,)   predicted labels
        proba : (n, K) classifier posterior probabilities

        After calling, check  self.drift[-1] == 2  for a drift signal.
        """
        self.chunk_idx += 1

        # Stage I — prequential accuracy update
        n = len(y)
        self.cum_correct += int(np.sum(pred == y))
        self.cum_total   += n
        p_i = self.cum_correct / self.cum_total
        s_i = np.sqrt(p_i * (1.0 - p_i) / self.cum_total)
        if p_i + s_i > self.p_max + self.s_max:
            self.p_max = p_i
            self.s_max = s_i

        # Buffer per-chunk statistics
        self.margin_buf.append(self._margins(proba))
        self.entropy_buf.append(self._entropy(proba))
        self.label_buf.append(y.copy())

        # Cooldown guard
        if self.cooldown > 0:
            self.cooldown -= 1
            self.drift.append(0)
            return self

        # Stage I candidate check
        is_candidate = (
            self.cum_total > 30
            and (p_i - s_i) < (self.p_max - self.lam * self.s_max)
        )
        if not is_candidate:
            self.consecutive = 0
            self.drift.append(0)
            return self

        # Stage II — multi-view validation
        w     = self.window_size
        T     = len(self.margin_buf) - 1
        s_neg = max(0, T - w)
        Wm = dict(
            margins=self.margin_buf[s_neg:T],
            entropy=self.entropy_buf[s_neg:T],
            labels=self.label_buf[s_neg:T],
        )
        Wp = dict(
            margins=self.margin_buf[T : min(T + w, len(self.margin_buf))],
            entropy=self.entropy_buf[T : min(T + w, len(self.entropy_buf))],
            labels=self.label_buf[T : min(T + w, len(self.label_buf))],
        )

        if not Wm["margins"] or not Wp["margins"]:
            self.drift.append(0)
            return self

        p_fuse = self._validate(Wm, Wp)
        self.consecutive = self.consecutive + 1 if p_fuse < self.alpha else 0

        if self.consecutive >= self.tau_p:
            self.drift.append(2)
            self.consecutive = 0
            self.cooldown    = self.tau_c
            # Reset Stage I after confirmed drift
            self.cum_correct = 0
            self.cum_total   = 0
            self.p_max       = 0.0
            self.s_max       = 0.0
        else:
            self.drift.append(0)

        return self


# ==============================================================================
# Baseline detectors (per-sample prequential updates)
# ==============================================================================

class DDM_stream:
    """DDM — Gama et al. (2004). Per-sample cumulative error monitoring."""

    def __init__(self, alpha_w=2.0, alpha_d=3.0, min_n=30):
        self.alpha_w = alpha_w
        self.alpha_d = alpha_d
        self.min_n   = min_n
        self.drift   = []
        self._reset()

    def _reset(self):
        self.p = 0.0; self.s = 0.0; self.n = 1
        self.p_min = np.inf; self.s_min = np.inf

    def feed(self, X, y, pred, proba=None):
        errors   = (pred != y).astype(int)
        detected = False
        for e in errors:
            self.p += (e - self.p) / self.n
            self.s  = np.sqrt(max(self.p * (1 - self.p) / self.n, 0.0))
            self.n += 1
            if self.n > self.min_n:
                if self.p + self.s <= self.p_min + self.s_min:
                    self.p_min = self.p; self.s_min = self.s
                if (self.s_min > 0 and
                        self.p + self.s > self.p_min + self.alpha_d * self.s_min):
                    self._reset(); detected = True; break
        if detected:
            self.drift.append(2)
        elif (self.n > self.min_n and self.s_min > 0 and
              self.p + self.s > self.p_min + self.alpha_w * self.s_min):
            self.drift.append(1)
        else:
            self.drift.append(0)
        return self


class EDDM_stream:
    """EDDM — Baena-García et al. (2006). Inter-error distance monitoring."""

    def __init__(self, alpha_w=0.95, alpha_d=0.90, min_errors=30):
        self.alpha_w = alpha_w
        self.alpha_d = alpha_d
        self.min_e   = min_errors
        self.drift   = []
        self._reset()

    def _reset(self):
        self.n = 0; self.last = 0; self.gaps = []
        self.m = 0.0; self.s = 0.0
        self.m_max = 0.0; self.s_max = 0.0

    def feed(self, X, y, pred, proba=None):
        for e in (pred != y).astype(int):
            self.n += 1
            if e == 1:
                self.gaps.append(self.n - self.last); self.last = self.n
                if len(self.gaps) >= 2:
                    self.m = float(np.mean(self.gaps))
                    self.s = float(np.std(self.gaps))
                    if self.m + 2 * self.s > self.m_max + 2 * self.s_max:
                        self.m_max = self.m; self.s_max = self.s
        denom = self.m_max + 2 * self.s_max
        if denom > 0 and len(self.gaps) >= self.min_e:
            ratio = (self.m + 2 * self.s) / denom
            if ratio < self.alpha_d:
                self.drift.append(2); self._reset(); return self
            if ratio < self.alpha_w:
                self.drift.append(1); return self
        self.drift.append(0)
        return self


class HDDM_A_stream:
    """HDDM-A — Frías-Blanco et al. (2014). Hoeffding bound on accuracy mean."""

    def __init__(self, delta_d=0.001, delta_w=0.005, min_n=30):
        self.dd = delta_d; self.dw = delta_w; self.m = min_n
        self.drift = []; self._reset()

    def _reset(self):
        self.n = 0; self.mu = 0.0; self.mu_max = 0.0

    def feed(self, X, y, pred, proba=None):
        for c in (pred == y).astype(float):
            self.n += 1; self.mu += (c - self.mu) / self.n
            if self.mu > self.mu_max: self.mu_max = self.mu
        if self.n > self.m:
            eps_d = np.sqrt(np.log(1 / self.dd) / (2 * self.n))
            eps_w = np.sqrt(np.log(1 / self.dw) / (2 * self.n))
            if self.mu_max - self.mu > eps_d:
                self.drift.append(2); self._reset(); return self
            if self.mu_max - self.mu > eps_w:
                self.drift.append(1); return self
        self.drift.append(0)
        return self


class HDDM_W_stream:
    """HDDM-W — Frías-Blanco et al. (2014). Hoeffding bound on EWMA."""

    def __init__(self, delta_d=0.001, delta_w=0.005, lam=0.05, min_n=30):
        self.dd = delta_d; self.dw = delta_w; self.lam = lam; self.m = min_n
        self.drift = []; self._reset()

    def _reset(self):
        self.n = 0; self.w = 0.0; self.w2 = 0.0
        self.wmax = 0.0; self.w2max = 0.0

    def feed(self, X, y, pred, proba=None):
        lam = self.lam
        for c in (pred == y).astype(float):
            self.n  += 1
            self.w   = lam * c     + (1 - lam) * self.w
            self.w2  = lam * c * c + (1 - lam) * self.w2
            if self.w > self.wmax: self.wmax = self.w; self.w2max = self.w2
        if self.n > self.m:
            var = max(self.w2 - self.w ** 2, 1e-12)
            ed  = np.sqrt(var * np.log(1 / self.dd) / 2)
            ew  = np.sqrt(var * np.log(1 / self.dw) / 2)
            if self.wmax - self.w > ed:
                self.drift.append(2); self._reset(); return self
            if self.wmax - self.w > ew:
                self.drift.append(1); return self
        self.drift.append(0)
        return self


class ADWIN_stream:
    """ADWIN — Bifet & Gavaldà (2007). Adaptive windowing on accuracy stream."""

    def __init__(self, delta=0.002):
        self.delta = delta; self.W = []; self.drift = []

    def _cut(self):
        n = len(self.W)
        if n < 5: return False
        total = sum(self.W)
        step  = max(1, int(np.sqrt(n)))
        for i in range(0, n - 1, step):
            s    = sum(self.W[: i + 1]); cnt0 = i + 1; cnt1 = n - cnt0
            if cnt1 < 1: break
            mu0  = s / cnt0; mu1 = (total - s) / cnt1
            var  = max(np.var(self.W), 1e-12); dp = self.delta / n
            m    = 1.0 / (1.0 / cnt0 + 1.0 / cnt1)
            cut  = (np.sqrt((2.0 / m) * var * np.log(2.0 / max(dp, 1e-15)))
                    + (2.0 / (3.0 * m)) * np.log(2.0 / max(dp, 1e-15)))
            if abs(mu0 - mu1) >= cut:
                del self.W[: i + 1]; return True
        return False

    def feed(self, X, y, pred, proba=None):
        self.W.append(float(np.mean(pred == y)))
        self.drift.append(2 if self._cut() else 0)
        return self


# ==============================================================================
# Evaluation metrics (paper-faithful)
# ==============================================================================

def calc_metrics(actual_drifts, detections, n_chunks, tolerance=3):
    """
    Compute TP, FP, FN, TPR, Precision, F1, EDDR, D1, D2.

    Matching rule: greedy closest-first within ±tolerance chunks.
    Each true drift can be matched at most once.

    EDDR (Eq. 5 in paper):
        EDDR = (TP / D) * 1 / (1 + FP / D)
    where D = len(actual_drifts).

    D1: mean distance from each FP detection to nearest true drift.
    D2: mean distance from each true drift to nearest TP detection.
    """
    actual   = list(actual_drifts)
    detected = list(detections)
    D        = len(actual)
    n_det    = len(detected)

    matched_drifts     = set()
    matched_detections = set()

    for di, det in enumerate(detected):
        if not actual: break
        dists = [abs(det - a) for a in actual]
        ci    = int(np.argmin(dists))
        if dists[ci] <= tolerance and ci not in matched_drifts:
            matched_drifts.add(ci)
            matched_detections.add(di)

    TP = len(matched_drifts)
    FP = n_det - len(matched_detections)
    FN = D - TP

    TPR       = TP / D       if D     > 0 else 0.0
    Precision = TP / n_det   if n_det > 0 else 0.0
    F1 = (2.0 * Precision * TPR / (Precision + TPR)
          if (Precision + TPR) > 0 else 0.0)

    # EDDR — paper Equation 5
    EDDR = (TP / D) * (1.0 / (1.0 + FP / D)) if D > 0 else 0.0

    # D1: FP detections → nearest true drift
    fp_dets = [det for di, det in enumerate(detected)
               if di not in matched_detections]
    D1 = (float(np.mean([min(abs(det - a) for a in actual) for det in fp_dets]))
          if fp_dets and actual else (float(n_chunks) if fp_dets else 0.0))

    # D2: each true drift → nearest TP detection
    tp_dets = [det for di, det in enumerate(detected)
               if di in matched_detections]
    D2 = (float(np.mean([
              min((abs(a - det) for det in tp_dets), default=n_chunks)
              for a in actual
          ])) if actual else 0.0)

    return dict(
        TP=TP, FP=FP, FN=FN,
        TPR=round(TPR, 4), Precision=round(Precision, 4),
        F1=round(F1, 4), EDDR=round(EDDR, 4),
        D1=round(D1, 4), D2=round(D2, 4),
    )


# ==============================================================================
# Experiment runner
# ==============================================================================

METHOD_NAMES = ["MV-Fuse", "DDM", "EDDM", "HDDM-A", "HDDM-W", "ADWIN"]
METRIC_KEYS  = ["TP", "FP", "FN", "TPR", "Precision", "F1", "EDDR", "D1", "D2"]


def _make_detectors():
    return {
        "MV-Fuse": MVFuse(window_size=5, alpha=0.05, tau_p=1, tau_c=5, lam=1.0),
        "DDM":     DDM_stream(),
        "EDDM":    EDDM_stream(),
        "HDDM-A":  HDDM_A_stream(),
        "HDDM-W":  HDDM_W_stream(),
        "ADWIN":   ADWIN_stream(),
    }


def run_experiment(
    n_chunks=60,
    chunk_size=200,
    drift_chunks=None,
    noise_levels=None,
    n_reps=5,
    tolerance=3,
    verbose=True,
):
    """
    Run the full benchmark experiment.

    Parameters
    ----------
    n_chunks     : chunks per stream
    chunk_size   : samples per chunk
    drift_chunks : indices of true drift points  (default [12,24,36,48])
    noise_levels : list of noise rates            (default [0,0.05,...,0.20])
    n_reps       : replications per condition
    tolerance    : ±chunk tolerance for TP matching
    verbose      : print per-condition tables

    Returns
    -------
    all_results : nested dict  [dataset][noise][method] → list of metric dicts
    overall     : dict [method] → {metric: list of all values}
    """
    if drift_chunks is None:
        drift_chunks = [12, 24, 36, 48]
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]

    all_results = {}

    for dname, gen in DATASETS.items():
        if verbose:
            print(f"\n{'='*65}\nDataset: {dname}\n{'='*65}")
        all_results[dname] = {}

        for noise in noise_levels:
            if verbose:
                print(f"\n  Noise: {noise*100:.0f}%")

            method_results = {m: [] for m in METHOD_NAMES}

            for rep in range(n_reps):
                np.random.seed(rep * 100 + 7)
                X_all, y_all = gen(n_chunks, chunk_size, drift_chunks, noise)
                classes      = np.unique(np.concatenate(y_all))

                detectors = _make_detectors()
                clfs      = {m: GaussianNB() for m in METHOD_NAMES}
                for m in METHOD_NAMES:
                    clfs[m].partial_fit(X_all[0], y_all[0], classes=classes)

                dets = {m: [] for m in METHOD_NAMES}

                for i in range(1, n_chunks):
                    X, y = X_all[i], y_all[i]
                    for nm, det in detectors.items():
                        pred  = clfs[nm].predict(X)
                        proba = clfs[nm].predict_proba(X)
                        det.feed(X, y, pred, proba)
                        if det.drift and det.drift[-1] == 2:
                            dets[nm].append(i)
                            clfs[nm] = GaussianNB()
                            clfs[nm].partial_fit(X, y, classes=classes)
                        else:
                            clfs[nm].partial_fit(X, y)

                for nm in METHOD_NAMES:
                    method_results[nm].append(
                        calc_metrics(drift_chunks, dets[nm], n_chunks, tolerance)
                    )

            all_results[dname][noise] = method_results

            if verbose:
                _print_table(method_results)

    # ── Overall summary ──────────────────────────────────────────────────
    overall = {m: {k: [] for k in METRIC_KEYS} for m in METHOD_NAMES}
    for d in DATASETS:
        for n in noise_levels:
            for m in METHOD_NAMES:
                for r in all_results[d][n][m]:
                    for k in METRIC_KEYS:
                        overall[m][k].append(r[k])

    if verbose:
        _print_overall(overall)
        _print_wilcoxon(overall)

    # ── Save results ─────────────────────────────────────────────────────
    with open("results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print("\nSaved: results.pkl")

    summary = {}
    for dname in DATASETS:
        summary[dname] = {}
        for noise in noise_levels:
            nk = f"{int(noise*100)}pct"
            summary[dname][nk] = {
                nm: {k: round(np.mean([r[k] for r in all_results[dname][noise][nm]]), 4)
                     for k in METRIC_KEYS}
                for nm in METHOD_NAMES
            }
    summary["OVERALL"] = {
        nm: {k: round(np.mean(overall[nm][k]), 4) for k in METRIC_KEYS}
        for nm in METHOD_NAMES
    }
    with open("results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved: results.json")

    return all_results, overall


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def _print_table(method_results):
    hdr = (f"  {'Method':<12} {'TP':>4} {'FP':>5} {'FN':>4} "
           f"{'TPR':>6} {'Prec':>6} {'F1':>6} {'EDDR':>6} {'D1':>6} {'D2':>6}")
    print(hdr)
    print(f"  {'-'*74}")
    for nm in METHOD_NAMES:
        avg = {k: np.mean([r[k] for r in method_results[nm]]) for k in METRIC_KEYS}
        print(f"  {nm:<12} {avg['TP']:>4.1f} {avg['FP']:>5.1f} {avg['FN']:>4.1f} "
              f"{avg['TPR']:>6.2f} {avg['Precision']:>6.2f} {avg['F1']:>6.3f} "
              f"{avg['EDDR']:>6.3f} {avg['D1']:>6.2f} {avg['D2']:>6.2f}")


def _print_overall(overall):
    print(f"\n{'='*75}")
    print("OVERALL SUMMARY (all datasets × all noise levels)")
    print(f"{'='*75}")
    print(f"{'Method':<12} {'TP':>5} {'FP':>6} {'FN':>5} "
          f"{'TPR':>7} {'Prec':>7} {'F1':>7} {'EDDR':>7} {'D1':>7} {'D2':>7}")
    print("-" * 75)
    for m in METHOD_NAMES:
        avg = {k: np.mean(overall[m][k]) for k in METRIC_KEYS}
        print(f"{m:<12} {avg['TP']:>5.1f} {avg['FP']:>6.1f} {avg['FN']:>5.1f} "
              f"{avg['TPR']:>7.2f} {avg['Precision']:>7.2f} {avg['F1']:>7.3f} "
              f"{avg['EDDR']:>7.3f} {avg['D1']:>7.2f} {avg['D2']:>7.2f}")


def _print_wilcoxon(overall):
    print(f"\n{'='*55}")
    print("Wilcoxon signed-rank test: MV-Fuse vs each baseline (F1)")
    print(f"{'='*55}")
    mv_f1 = overall["MV-Fuse"]["F1"]
    print(f"{'Comparison':<28} {'W-stat':>8} {'p-value':>10}  Sig")
    print("-" * 55)
    for bl in ["DDM", "EDDM", "HDDM-A", "HDDM-W", "ADWIN"]:
        diff = [m - b for m, b in zip(mv_f1, overall[bl]["F1"])]
        try:
            stat, pval = wilcoxon(diff, alternative="greater")
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else
                  ("*" if pval < 0.05 else "ns"))
            print(f"MV-Fuse vs {bl:<16} {stat:>8.1f} {pval:>10.4f}  {sig}")
        except Exception as e:
            print(f"MV-Fuse vs {bl:<16}  error: {e}")

    f1s = sorted(
        [(m, np.mean(overall[m]["F1"])) for m in METHOD_NAMES],
        key=lambda x: -x[1],
    )
    print("\nFinal ranking by F1:")
    medals = ["1.", "2.", "3."]
    for i, (m, f1) in enumerate(f1s, 1):
        print(f"  {medals[i-1] if i <= 3 else '  '} {m:<12}  F1={f1:.3f}")


# ==============================================================================
if __name__ == "__main__":
    run_experiment(n_reps=5, verbose=True)
