"""Step 3. Run the trained model on a split and save per-image probabilities.

    python scripts/03_predict.py --model models/best_model.pth \
        --split-csv data/splits/test.csv --images data/images --out results/predictions.csv

Output columns: image, patient_id, y_true, y_prob (probability of malignancy),
y_pred (argmax class), plus age and gender when present. This file contains
patient identifiers. Keep it out of version control.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from eyelid_audit.data import TumorDataset, build_transforms
from eyelid_audit.model import load_trained_model
from eyelid_audit.utils import ensure_dir, get_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    model = load_trained_model(args.model, device)
    ds = TumorDataset(args.split_csv, args.images, build_transforms(train=False), return_name=True)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    names, y_true, y_prob, y_pred = [], [], [], []
    with torch.no_grad():
        for imgs, labels, fnames in loader:
            logits = model(imgs.to(device))
            y_prob += torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            y_pred += torch.argmax(logits, dim=1).cpu().tolist()
            y_true += labels.tolist()
            names += list(fnames)

    # The loader is not shuffled, so rows line up with the split CSV.
    if names != ds.df["image"].tolist():
        sys.exit("Row order mismatch between loader and split CSV.")
    out = pd.DataFrame({"image": names})
    if "patient_id" in ds.df.columns:
        out["patient_id"] = ds.df["patient_id"].values
    out["y_true"], out["y_prob"], out["y_pred"] = y_true, y_prob, y_pred
    for col in ("age", "gender"):
        if col in ds.df.columns:
            out[col] = ds.df[col].values

    ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    out.to_csv(args.out, index=False)
    acc = (out["y_true"] == out["y_pred"]).mean()
    print(f"Wrote {len(out)} predictions to {args.out} (accuracy {acc:.3f})")


if __name__ == "__main__":
    main()
