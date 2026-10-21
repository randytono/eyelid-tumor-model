"""Step 7. Clustering-aware statistics for localization failures.

    python scripts/07_localization_stats.py --predictions results/predictions.csv \
        --labels results/localization_labels.csv --rater rater1_miss --out results

Images are nested in patients, so every model here accounts for clustering:
- population-average failure rate: intercept-only GEE (logit link,
  exchangeable working correlation, patient clusters), plus a patient-cluster
  bootstrap of the image-level rate as a non-parametric check;
- associations with malignancy, age and sex: GEE logistic models;
- localization failure versus classification error: GEE and a 2x2 table.

With few patients (22 in the paper), robust GEE standard errors can be
slightly optimistic and sparse categories can separate. Estimates flagged as
unstable should not be reported.
"""

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial

from eyelid_audit import config
from eyelid_audit.metrics import bootstrap_rate
from eyelid_audit.utils import ensure_dir, normalize_sex


def expit(v):
    return 1.0 / (1.0 + np.exp(-v))


def fit_gee(formula, data):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = smf.gee(formula, groups="patient_id", data=data,
                      family=Binomial(), cov_struct=Exchangeable()).fit()
    return res, [str(w.message) for w in caught]


def odds_ratios(res):
    rows = []
    ci = res.conf_int()
    for name in res.params.index:
        if name == "Intercept":
            continue
        b, lo, hi = res.params[name], ci.loc[name, 0], ci.loc[name, 1]
        unstable = (not np.all(np.isfinite([lo, hi]))) or abs(b) > 10 or (hi - lo) > 20
        with np.errstate(over="ignore"):
            rows.append({"term": name, "or": float(np.exp(b)), "ci95": [float(np.exp(lo)), float(np.exp(hi))],
                         "p": float(res.pvalues[name]), "unstable": bool(unstable)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--labels", required=True, help="CSV with 'image' and a 0/1 failure column")
    ap.add_argument("--rater", default="rater1_miss", help="Failure column used as the primary label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-resamples", type=int, default=config.BOOTSTRAP_RESAMPLES)
    ap.add_argument("--seed", type=int, default=config.BOOTSTRAP_SEED)
    args = ap.parse_args()
    ensure_dir(args.out)

    pred = pd.read_csv(args.predictions)
    labels = pd.read_csv(args.labels)
    if args.rater not in labels.columns:
        sys.exit(f"Labels file has no column '{args.rater}'.")
    df = pred.merge(labels[["image", args.rater]], on="image", how="inner")
    if len(df) != len(pred):
        missing = sorted(set(pred["image"]) - set(df["image"]))
        sys.exit(f"{len(missing)} predicted images have no localization label, e.g. {missing[:5]}")

    df["loc_fail"] = pd.to_numeric(df[args.rater]).astype(int)
    df["class_error"] = (df["y_pred"] != df["y_true"]).astype(int)
    df["malignant"] = df["y_true"].astype(int)
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "gender" in df.columns:
        df["sex"] = df["gender"].map(normalize_sex)

    n, n_pat, n_fail = len(df), df["patient_id"].nunique(), int(df["loc_fail"].sum())
    per_patient = df.groupby("patient_id").agg(n_images=("image", "size"), n_failures=("loc_fail", "sum"))
    pats_with_fail = int((per_patient["n_failures"] > 0).sum())
    per_patient.sort_values("n_failures", ascending=False).to_csv(
        os.path.join(args.out, "failures_per_patient.csv"))

    out = {"label_column": args.rater, "n_images": n, "n_patients": int(n_pat), "n_failures": n_fail,
           "image_level_rate": n_fail / n, "patients_with_failure": pats_with_fail}
    lines = [f"Label column: {args.rater}",
             f"Localization failures: {n_fail} of {n} images ({n_fail / n:.1%}), "
             f"in {pats_with_fail} of {n_pat} patients",
             f"Failures per affected patient: "
             f"{sorted(per_patient.loc[per_patient['n_failures'] > 0, 'n_failures'].astype(int).tolist(), reverse=True)}"]

    res0, warn0 = fit_gee("loc_fail ~ 1", df)
    b0, ci0 = res0.params["Intercept"], res0.conf_int().loc["Intercept"]
    icc = float(np.atleast_1d(res0.cov_struct.dep_params)[0])
    out["gee_rate"] = {"rate": float(expit(b0)), "ci95": [float(expit(ci0[0])), float(expit(ci0[1]))],
                       "within_patient_correlation": icc, "warnings": warn0}
    rng = np.random.default_rng(args.seed)
    rate, rate_ci = bootstrap_rate(df["loc_fail"].to_numpy(), df["patient_id"].to_numpy(), rng, args.n_resamples)
    out["bootstrap_rate"] = {"rate": rate, "ci95": list(rate_ci), "n_resamples": args.n_resamples, "seed": args.seed}
    lines += ["", "FAILURE RATE",
              f"  GEE population-average: {expit(b0):.1%} (95% CI {expit(ci0[0]):.1%} to {expit(ci0[1]):.1%}); "
              f"within-patient correlation {icc:.2f}",
              f"  Patient-cluster bootstrap of image-level rate: {rate:.1%} "
              f"(95% CI {rate_ci[0]:.1%} to {rate_ci[1]:.1%})"]

    models = [("Malignant vs benign", "loc_fail ~ C(malignant)")]
    if "age" in df.columns and df["age"].notna().any():
        models.append(("Age (per year)", "loc_fail ~ age"))
    if "sex" in df.columns and df["sex"].nunique() > 1:
        models.append(("Male vs female", "loc_fail ~ C(sex, Treatment(reference='female'))"))
    models.append(("Classification error", "loc_fail ~ C(class_error)"))

    out["associations"] = {}
    lines += ["", "ASSOCIATIONS WITH LOCALIZATION FAILURE (GEE, patient clusters)"]
    for label, formula in models:
        data = df.dropna(subset=["age"]) if "age" in formula else df
        try:
            res, warns = fit_gee(formula, data)
            rows = odds_ratios(res)
        except Exception as exc:  # e.g. perfect separation
            out["associations"][label] = {"error": str(exc)}
            lines.append(f"  {label}: could not be estimated ({exc})")
            continue
        out["associations"][label] = {"formula": formula, "terms": rows, "warnings": warns}
        for r in rows:
            flag = "  UNSTABLE, do not report" if r["unstable"] else ""
            lines.append(f"  {label}: OR {r['or']:.2f} (95% CI {r['ci95'][0]:.2f} to {r['ci95'][1]:.2f}), "
                         f"p = {r['p']:.3f}{flag}")

    table = pd.crosstab(df["loc_fail"], df["class_error"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    fail_on_correct = int(table.loc[1, 0])
    out["failure_vs_classification"] = {
        "table_rows_loc_fail_cols_class_error": table.values.tolist(),
        "failures_on_correct_predictions": fail_on_correct,
        "share_of_failures_on_correct_predictions": fail_on_correct / n_fail if n_fail else None}
    lines += ["", "LOCALIZATION FAILURE x CLASSIFICATION ERROR",
              "  rows: localization failure (0/1), columns: classification error (0/1)",
              "  " + table.to_string().replace("\n", "\n  ")]
    if n_fail:
        lines.append(f"  {fail_on_correct} of {n_fail} failures ({fail_on_correct / n_fail:.0%}) "
                     f"occurred on correctly classified images")

    with open(os.path.join(args.out, "localization_stats.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    report = "\n".join(lines)
    with open(os.path.join(args.out, "localization_report.txt"), "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved localization_stats.json, localization_report.txt and failures_per_patient.csv to {args.out}")


if __name__ == "__main__":
    main()
