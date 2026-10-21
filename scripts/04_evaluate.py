"""Step 4. Diagnostic performance with patient-cluster bootstrap confidence intervals.

    python scripts/04_evaluate.py --predictions results/predictions.csv \
        --splits data/splits --out results

Reports accuracy, sensitivity, specificity, AUC, average precision and Brier
score at the 0.5 threshold, each with a 95% CI from 5000 patient-level
bootstrap resamples (seed 42). Also reports subgroup AUCs with their sizes,
the train-versus-test comparability tests of Table 1 (when --splits is given),
and saves ROC, precision-recall, calibration, confusion-matrix and age plots.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, gaussian_kde, ttest_ind
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from eyelid_audit import config
from eyelid_audit.metrics import (METRIC_ORDER, binary_metrics, bootstrap_metrics,
                                  bootstrap_subgroup_auc)
from eyelid_audit.utils import ensure_dir, normalize_sex


def fmt_ci(point, ci, pct=False):
    if point is None or (isinstance(point, float) and np.isnan(point)):
        return "undefined"
    if pct:
        return f"{point:.1%} (95% CI {ci[0]:.1%} to {ci[1]:.1%})"
    return f"{point:.3f} (95% CI {ci[0]:.3f} to {ci[1]:.3f})"


def comparability_tests(splits_dir):
    """Table 1: training versus test set.

    Age (Welch t-test) and sex (chi-square) are compared at the patient level;
    the malignant proportion (chi-square) at the image level.
    """
    train = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    test = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    out = {"train_images": len(train), "test_images": len(test)}

    tr_pat = train.drop_duplicates(subset=["patient_id"])
    te_pat = test.drop_duplicates(subset=["patient_id"])
    out["train_patients"], out["test_patients"] = len(tr_pat), len(te_pat)

    age_tr = pd.to_numeric(tr_pat["age"], errors="coerce").dropna()
    age_te = pd.to_numeric(te_pat["age"], errors="coerce").dropna()
    out["age_train_mean_sd"] = [float(age_tr.mean()), float(age_tr.std())]
    out["age_test_mean_sd"] = [float(age_te.mean()), float(age_te.std())]
    out["age_p_welch_t"] = float(ttest_ind(age_tr, age_te, equal_var=False).pvalue)

    if "gender" in train.columns:
        sex_tr = tr_pat["gender"].map(normalize_sex)
        sex_te = te_pat["gender"].map(normalize_sex)
        table = [[(sex_tr == "male").sum(), (sex_tr == "female").sum()],
                 [(sex_te == "male").sum(), (sex_te == "female").sum()]]
        out["sex_counts_male_female"] = {"train": [int(v) for v in table[0]],
                                         "test": [int(v) for v in table[1]]}
        out["sex_p_chi2"] = float(chi2_contingency(table)[1])

    cls_table = [[int((train["target"] == 0).sum()), int((train["target"] == 1).sum())],
                 [int((test["target"] == 0).sum()), int((test["target"] == 1).sum())]]
    out["class_counts_benign_malignant"] = {"train": cls_table[0], "test": cls_table[1]}
    out["class_p_chi2"] = float(chi2_contingency(cls_table)[1])
    return out, age_tr, age_te


def save_plots(df, cis, out_dir, ages=None):
    y, p = df["y_true"].to_numpy(), df["y_prob"].to_numpy()

    fpr, tpr, _ = roc_curve(y, p)
    auc = binary_metrics(y, p)["auc"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2,
            label=f"ResNet-50 (AUC = {auc:.3f})\n95% CI {cis['auc'][0]:.3f} to {cis['auc'][1]:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance (AUC = 0.50)")
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
           title="Receiver operating characteristic", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(rec, prec, where="post", lw=2, color="purple",
            label=f"AP = {binary_metrics(y, p)['ap']:.3f}")
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title="Precision-recall")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out_dir, "precision_recall_curve.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, "o-", label=f"ResNet-50 (Brier = {binary_metrics(y, p)['brier']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction malignant",
           title="Calibration (10 equal-width bins)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out_dir, "calibration_curve.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    y_pred = (p >= config.THRESHOLD).astype(int)
    cm = confusion_matrix(y, y_pred, labels=[1, 0])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["Malignant", "Benign"]); ax.set_yticks([0, 1], ["Malignant", "Benign"])
    ax.set(xlabel="Model prediction", ylabel="Histopathology", title="Confusion matrix (threshold 0.5)")
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    if ages is not None:
        age_tr, age_te = ages
        grid = np.linspace(min(age_tr.min(), age_te.min()) - 15, max(age_tr.max(), age_te.max()) + 15, 300)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for vals, name in ((age_tr, "Training set"), (age_te, "Test set")):
            dens = gaussian_kde(vals)(grid)
            ax.fill_between(grid, dens, alpha=0.3); ax.plot(grid, dens, label=name)
        ax.set(xlabel="Age (years)", ylabel="Density", title="Age distribution (patient level)")
        ax.legend()
        fig.savefig(os.path.join(out_dir, "age_distribution.png"), dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="Output of 03_predict.py")
    ap.add_argument("--splits", default=None, help="Folder with train.csv and test.csv (for Table 1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=config.THRESHOLD)
    ap.add_argument("--n-resamples", type=int, default=config.BOOTSTRAP_RESAMPLES)
    ap.add_argument("--seed", type=int, default=config.BOOTSTRAP_SEED)
    args = ap.parse_args()
    ensure_dir(args.out)

    df = pd.read_csv(args.predictions)
    for col in ("y_true", "y_prob", "patient_id"):
        if col not in df.columns:
            sys.exit(f"Predictions file needs a '{col}' column.")
    y = df["y_true"].astype(int).to_numpy()
    p = df["y_prob"].astype(float).to_numpy()
    pid = df["patient_id"].to_numpy()
    if set(np.unique(p)) <= {0.0, 1.0}:
        sys.exit("y_prob looks binary. AUC, AP and Brier need the continuous probability.")

    rng = np.random.default_rng(args.seed)
    lines = [f"{len(df)} images from {len(np.unique(pid))} patients, "
             f"{y.mean():.1%} malignant, threshold {args.threshold}",
             f"Patient-cluster bootstrap: {args.n_resamples} resamples, seed {args.seed}", ""]

    point = binary_metrics(y, p, args.threshold)
    cis, single = bootstrap_metrics(y, p, pid, rng, args.n_resamples, args.threshold)
    result = {"n_images": int(len(df)), "n_patients": int(len(np.unique(pid))),
              "threshold": args.threshold, "metrics": {}}
    lines.append("HEADLINE METRICS")
    for k in METRIC_ORDER:
        result["metrics"][k] = {"point": point[k], "ci95": list(cis[k])}
        pct = k in ("accuracy", "sensitivity", "specificity")
        lines.append(f"  {k:<12} {fmt_ci(point[k], cis[k], pct=pct)}")
    if single:
        lines.append(f"  ({single} resamples held one class only; excluded for AUC, AP, sensitivity, specificity)")

    fpr, tpr, thr = roc_curve(y, p)
    j = int(np.argmax(tpr - fpr))
    result["youden"] = {"threshold": float(thr[j]), "sensitivity": float(tpr[j]), "specificity": float(1 - fpr[j])}
    lines.append(f"  Youden-optimal threshold {thr[j]:.3f}: sensitivity {tpr[j]:.3f}, "
                 f"specificity {1 - fpr[j]:.3f} (reference only)")

    # Subgroups: same order as the published analysis (age <50, age >=50, then sex).
    lines += ["", "SUBGROUP AUC (report n with every estimate)"]
    result["subgroups"] = {}
    masks = []
    if "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce").to_numpy()
        masks += [("age < 50", age < 50), ("age >= 50", age >= 50)]
    if "gender" in df.columns:
        sex = df["gender"].map(normalize_sex).to_numpy()
        masks += [(f"sex = {lvl}", sex == lvl) for lvl in np.unique(sex)]
    for name, mask in masks:
        yy, pp, gg = y[mask], p[mask], pid[mask]
        auc, ci = bootstrap_subgroup_auc(yy, pp, gg, rng, args.n_resamples)
        entry = {"n_images": int(mask.sum()), "n_patients": int(len(np.unique(gg))),
                 "n_malignant": int(yy.sum()), "n_benign": int((1 - yy).sum()),
                 "auc": auc, "ci95": list(ci)}
        result["subgroups"][name] = entry
        note = ""
        if auc is None:
            note = "  AUC undefined (one class only)"
        elif auc >= 0.9999:
            note = "  perfectly separable small subgroup, not evidence of flawless performance"
        lines.append(f"  {name:<14} n={entry['n_images']} images / {entry['n_patients']} patients "
                     f"({entry['n_malignant']} malignant): AUC {fmt_ci(auc, ci) if auc is not None else '-'}{note}")

    ages = None
    if args.splits:
        comp, age_tr, age_te = comparability_tests(args.splits)
        ages = (age_tr, age_te)
        result["comparability"] = comp
        lines += ["", "TRAIN VS TEST COMPARABILITY (Table 1)",
                  f"  Images {comp['train_images']} vs {comp['test_images']}; "
                  f"patients {comp['train_patients']} vs {comp['test_patients']}",
                  f"  Age (patient level, Welch t-test) p = {comp['age_p_welch_t']:.4f}",
                  f"  Malignant proportion (image level, chi-square) p = {comp['class_p_chi2']:.4f}"]
        if "sex_p_chi2" in comp:
            lines.append(f"  Sex (patient level, chi-square) p = {comp['sex_p_chi2']:.4f}")

    save_plots(df, cis, args.out, ages)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    report = "\n".join(lines)
    with open(os.path.join(args.out, "metrics_report.txt"), "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved metrics.json, metrics_report.txt and figures to {args.out}")


if __name__ == "__main__":
    main()
