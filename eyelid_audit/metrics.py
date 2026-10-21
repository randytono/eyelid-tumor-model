"""Diagnostic metrics and patient-cluster bootstrap.

The 98 test images come from 22 patients, so images are not independent. All
confidence intervals resample patients, not images.
"""

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)

METRIC_ORDER = ["accuracy", "sensitivity", "specificity", "auc", "ap", "brier"]


def binary_metrics(y_true, y_prob, threshold=0.5):
    """Accuracy, sensitivity, specificity, AUC, AP and Brier score.

    Metrics that need both classes return NaN when only one class is present.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": float((y_pred == y_true).mean()),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "auc": np.nan, "ap": np.nan, "sensitivity": np.nan, "specificity": np.nan,
    }
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["ap"] = float(average_precision_score(y_true, y_prob))
        out["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
        out["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    return out


def patient_index(groups):
    """Sorted unique patients and the row indices belonging to each."""
    groups = np.asarray(groups)
    patients = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in patients}
    return patients, rows


def cluster_resample_indices(rng, patients, rows):
    """One bootstrap sample: draw patients with replacement, keep all their images."""
    chosen = rng.choice(patients, size=len(patients), replace=True)
    return np.concatenate([rows[c] for c in chosen])


def percentile_ci(values, level=0.95):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (np.nan, np.nan)
    tail = (1 - level) / 2 * 100
    return (float(np.percentile(arr, tail)), float(np.percentile(arr, 100 - tail)))


def bootstrap_metrics(y_true, y_prob, groups, rng, n_resamples, threshold=0.5):
    """Patient-cluster bootstrap CIs for all headline metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    patients, rows = patient_index(groups)
    draws = {k: [] for k in METRIC_ORDER}
    single_class = 0
    for _ in range(n_resamples):
        idx = cluster_resample_indices(rng, patients, rows)
        m = binary_metrics(y_true[idx], y_prob[idx], threshold)
        if np.isnan(m["auc"]):
            single_class += 1
        for k in METRIC_ORDER:
            draws[k].append(m[k])
    cis = {k: percentile_ci(draws[k]) for k in METRIC_ORDER}
    return cis, single_class


def bootstrap_subgroup_auc(y_true, y_prob, groups, rng, n_resamples):
    """AUC with a patient-cluster bootstrap CI inside one subgroup.

    Returns None for the AUC when the subgroup holds only one class.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None, (np.nan, np.nan)
    point = float(roc_auc_score(y_true, y_prob))
    patients, rows = patient_index(groups)
    aucs = []
    for _ in range(n_resamples):
        idx = cluster_resample_indices(rng, patients, rows)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return point, percentile_ci(aucs)


def bootstrap_rate(flags, groups, rng, n_resamples):
    """Patient-cluster bootstrap CI for a proportion (e.g. localization failure rate)."""
    flags = np.asarray(flags, dtype=float)
    patients, rows = patient_index(groups)
    rates = [flags[cluster_resample_indices(rng, patients, rows)].mean()
             for _ in range(n_resamples)]
    return float(flags.mean()), percentile_ci(rates)
