"""Synthetic stand-in data so the pipeline can be run without patient images.

    python scripts/make_synthetic_data.py images --out demo/data
    python scripts/make_synthetic_data.py raters --sheet demo/gradcam/rater_sheet.csv --out demo/raters

`images` draws random skin-tone pictures with one coloured blob whose shade
depends on the class, and writes metadata.csv with the same columns as the real
study. Patient IDs look like SYN-0001 and are not real.

`raters` fills two rater sheets with random scores (rater 2 slightly stricter),
only to exercise the kappa and GEE steps.

Results on synthetic data are meaningless. They test the code, not the findings.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


def make_image(rng, malignant, size=256):
    base = np.array([205, 160, 135]) + rng.normal(0, 12, 3)
    img = np.clip(base + rng.normal(0, 10, (size, size, 3)), 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    cx, cy = rng.integers(70, size - 70, 2)
    rx, ry = rng.integers(18, 45, 2)
    color = (120, 40, 45) if malignant else (150, 105, 80)
    color = tuple(int(np.clip(c + rng.normal(0, 15), 0, 255)) for c in color)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
    return pil.filter(ImageFilter.GaussianBlur(1.2))


def cmd_images(args):
    rng = np.random.default_rng(args.seed)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows, counter = [], 1
    for p in range(1, args.n_patients + 1):
        malignant = rng.random() < 0.62
        age = int(np.clip(rng.normal(55, 18), 5, 92))
        sex = rng.choice(["m", "f"])
        for _ in range(int(rng.integers(1, 9))):
            name = f"{counter:04d}.jpg"
            make_image(rng, malignant).save(os.path.join(img_dir, name), quality=90)
            rows.append({"image": name, "patient_id": f"SYN-{p:04d}",
                         "label": "malignant" if malignant else "benign", "age": age, "gender": sex})
            counter += 1
    meta = pd.DataFrame(rows)
    meta.to_csv(os.path.join(args.out, "metadata.csv"), index=False)
    print(f"Wrote {len(meta)} synthetic images from {args.n_patients} patients to {args.out}")


def cmd_raters(args):
    rng = np.random.default_rng(args.seed)
    sheet = pd.read_csv(args.sheet)
    r1 = (rng.random(len(sheet)) < 0.3).astype(int)
    r2 = r1.copy()
    r2[(r1 == 0) & (rng.random(len(sheet)) < 0.15)] = 1  # rater 2 slightly stricter
    os.makedirs(args.out, exist_ok=True)
    for name, scores in (("rater1.csv", r1), ("rater2.csv", r2)):
        filled = sheet.copy()
        filled["miss"] = scores
        filled.to_csv(os.path.join(args.out, name), index=False)
    print(f"Wrote synthetic rater1.csv and rater2.csv to {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("images")
    a.add_argument("--out", required=True)
    a.add_argument("--n-patients", type=int, default=60)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=cmd_images)
    b = sub.add_parser("raters")
    b.add_argument("--sheet", required=True, help="rater_sheet.csv from 05_gradcam.py gallery")
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=1)
    b.set_defaults(func=cmd_raters)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
