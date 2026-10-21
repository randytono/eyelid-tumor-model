"""Step 6. Inter-rater agreement on localization scoring (Cohen's kappa).

Each rater fills a copy of rater_sheet.csv (from 05_gradcam.py gallery) with
miss = 1 (activation peak off the lesion) or 0 (on the lesion).

    python scripts/06_interrater_kappa.py --rater1 rater1.csv --rater2 rater2.csv \
        --out results/localization_labels.csv

Writes a merged labels file (image, rater1_miss, rater2_miss) for step 7.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from eyelid_audit.utils import ensure_dir

LANDIS_KOCH = [(0.81, "almost perfect"), (0.61, "substantial"), (0.41, "moderate"),
               (0.21, "fair"), (0.0, "slight"), (-1.0, "poor")]


def read_scores(path, column):
    df = pd.read_csv(path)
    if "image" not in df.columns or column not in df.columns:
        sys.exit(f"{path} needs columns 'image' and '{column}'.")
    vals = pd.to_numeric(df[column], errors="coerce")
    bad = df.loc[~vals.isin([0, 1]), "image"].tolist()
    if bad:
        sys.exit(f"{path}: '{column}' must be 0 or 1 for every row. Check: {bad[:10]}")
    return pd.DataFrame({"image": df["image"], "miss": vals.astype(int)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rater1", required=True)
    ap.add_argument("--rater2", required=True)
    ap.add_argument("--column", default="miss", help="Score column in both sheets")
    ap.add_argument("--out", required=True, help="Merged labels CSV")
    args = ap.parse_args()

    r1, r2 = read_scores(args.rater1, args.column), read_scores(args.rater2, args.column)
    merged = r1.merge(r2, on="image", suffixes=("_1", "_2"), how="inner")
    if len(merged) != len(r1) or len(merged) != len(r2):
        sys.exit(f"Image lists differ: rater1 {len(r1)}, rater2 {len(r2)}, shared {len(merged)}.")
    a, b = merged["miss_1"].to_numpy(), merged["miss_2"].to_numpy()

    kappa = cohen_kappa_score(a, b)
    agree = (a == b).mean()
    label = next(name for cut, name in LANDIS_KOCH if kappa >= cut)
    cm = confusion_matrix(a, b, labels=[0, 1])

    print(f"Images scored: {len(merged)}")
    print(f"Failures: rater 1 = {a.sum()}, rater 2 = {b.sum()}")
    print(f"Raw agreement: {agree:.1%}")
    print(f"Cohen's kappa: {kappa:.3f} ({label} agreement, Landis and Koch)")
    print("\nRows rater 1, columns rater 2 (0 = on lesion, 1 = failure):")
    print(pd.DataFrame(cm, index=["r1 = 0", "r1 = 1"], columns=["r2 = 0", "r2 = 1"]))
    print(f"\nDisagreements: rater 2 stricter on {cm[0, 1]}, rater 1 stricter on {cm[1, 0]}.")

    ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    merged.rename(columns={"miss_1": "rater1_miss", "miss_2": "rater2_miss"}).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
