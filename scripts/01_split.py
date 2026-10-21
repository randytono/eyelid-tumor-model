"""Step 1. Patient-level train/validation/test split with demographic balancing.

No patient contributes images to more than one partition (GroupShuffleSplit on
patient_id). Among candidate random splits, the script keeps the one whose
development and test sets are most similar in age (highest Kolmogorov-Smirnov
p-value), provided the malignant proportion differs by at most 15 percentage
points. Model performance is never looked at during selection.

Reproduce the published split exactly:
    python scripts/01_split.py --metadata data/metadata.csv --out data/splits --seed 696

Rerun the search:
    python scripts/01_split.py --metadata data/metadata.csv --out data/splits --n-trials 1000

The result depends on the row order of the metadata file, so keep it unchanged.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import GroupShuffleSplit

from eyelid_audit import config
from eyelid_audit.utils import ensure_dir


def load_metadata(path):
    df = pd.read_csv(path)
    required = {"image", "patient_id", "label", "age"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Metadata is missing columns: {sorted(missing)}")
    df["age_numeric"] = pd.to_numeric(df["age"], errors="coerce")
    df["target"] = df["label"].astype(str).str.strip().str.lower().map(config.LABEL_MAP)
    return df.dropna(subset=["target", "image"]).copy()


def split_for_seed(df, seed):
    """Test split (15% of patients), then validation split (20% of the rest)."""
    gss_test = GroupShuffleSplit(n_splits=1, test_size=config.SPLIT_TEST_SIZE, random_state=seed)
    dev_idx, test_idx = next(gss_test.split(df, df["target"], groups=df["patient_id"]))
    df_dev, df_test = df.iloc[dev_idx], df.iloc[test_idx]

    ages_dev, ages_test = df_dev["age_numeric"].dropna(), df_test["age_numeric"].dropna()
    p_age = ks_2samp(ages_dev, ages_test).pvalue if len(ages_dev) and len(ages_test) else 0.0
    ratio_diff = abs(df_dev["target"].mean() - df_test["target"].mean())
    score = 0.0 if ratio_diff > config.SPLIT_MAX_RATIO_DIFF else p_age

    gss_val = GroupShuffleSplit(n_splits=1, test_size=config.SPLIT_VAL_SIZE, random_state=seed)
    train_idx, val_idx = next(gss_val.split(df_dev, df_dev["target"], groups=df_dev["patient_id"]))
    return {
        "seed": int(seed), "score": float(score), "ks_p_age": float(p_age),
        "malignant_ratio_diff": float(ratio_diff),
        "train": df_dev.iloc[train_idx], "val": df_dev.iloc[val_idx], "test": df_test,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, help="CSV with image, patient_id, label, age (and gender)")
    ap.add_argument("--out", required=True, help="Folder for train.csv, val.csv, test.csv")
    ap.add_argument("--seed", type=int, default=None, help="Use this seed directly instead of searching")
    ap.add_argument("--n-trials", type=int, default=config.SPLIT_N_TRIALS, help="Seeds 0..n-1 to search")
    args = ap.parse_args()

    df = load_metadata(args.metadata)
    print(f"{len(df)} labelled images from {df['patient_id'].nunique()} patients")

    if args.seed is not None:
        best = split_for_seed(df, args.seed)
    else:
        best = None
        for seed in range(args.n_trials):
            cand = split_for_seed(df, seed)
            if best is None or cand["score"] > best["score"]:  # first maximum wins
                best = cand

    ensure_dir(args.out)
    for name in ("train", "val", "test"):
        best[name].to_csv(os.path.join(args.out, f"{name}.csv"), index=False)

    report = {k: best[k] for k in ("seed", "score", "ks_p_age", "malignant_ratio_diff")}
    report["n_trials_searched"] = None if args.seed is not None else args.n_trials
    for name in ("train", "val", "test"):
        part = best[name]
        report[f"{name}_images"] = int(len(part))
        report[f"{name}_patients"] = int(part["patient_id"].nunique())
        report[f"{name}_malignant_fraction"] = float(part["target"].mean())
    with open(os.path.join(args.out, "split_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"Selected seed {report['seed']}: age KS p = {report['ks_p_age']:.4f}, "
          f"malignant proportion gap = {report['malignant_ratio_diff']:.1%}")
    print(f"Train {report['train_images']} | Val {report['val_images']} | Test {report['test_images']} images")
    print(f"Wrote splits to {args.out}")


if __name__ == "__main__":
    main()
