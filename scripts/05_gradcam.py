"""Step 5. Grad-CAM++ attention maps for the faithfulness audit.

Grad-CAM++ is computed on the final convolutional block (layer4) for the
predicted class, as in the paper.

Blinded gallery for raters (original photo next to the overlay, no prediction,
no confidence, shuffled order) plus an empty scoring sheet:
    python scripts/05_gradcam.py gallery --model models/best_model.pth \
        --split-csv data/splits/test.csv --images data/images --out results/gradcam

Layer-progression figure for one image (layer3, layer4 block 2, final block):
    python scripts/05_gradcam.py progression --model models/best_model.pth \
        --images data/images --image 0690.jpg --out results/gradcam

Scoring rule (see docs/rater_instructions.md): a localization FAILURE is
recorded when the point of maximum activation does not fall on the lesion.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from eyelid_audit import config
from eyelid_audit.data import build_transforms, denormalize, load_rgb
from eyelid_audit.model import load_trained_model
from eyelid_audit.utils import ensure_dir, get_device, stem

RULE = ("Rule: does the point of maximum activation (deepest red) fall on the lesion?\n"
        "Score 1 = no (failure), 0 = yes.")


def prepare(image_path, transform, device):
    tensor = transform(image=load_rgb(image_path))["image"]
    return tensor.unsqueeze(0).to(device), denormalize(tensor)


def predict(model, x):
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    pred = int(torch.argmax(probs).item())
    return pred, float(probs[1].item())


def cam_map(model, layer, x, target_class):
    with GradCAMPlusPlus(model=model, target_layers=[layer]) as cam:
        return cam(input_tensor=x, targets=[ClassifierOutputTarget(target_class)])[0]


def gallery(args):
    device = get_device(args.device)
    model = load_trained_model(args.model, device)
    transform = build_transforms(train=False)
    df = pd.read_csv(args.split_csv)
    if args.no_shuffle:
        order = df.reset_index(drop=True)
    else:
        order = df.sample(frac=1.0, random_state=args.shuffle_seed).reset_index(drop=True)
    out_dir = ensure_dir(args.out)

    records = []
    for i, row in order.iterrows():
        x, rgb = prepare(os.path.join(args.images, row["image"]), transform, device)
        pred, p_mal = predict(model, x)
        heat = cam_map(model, model.layer4[-1], x, pred)
        py, px = np.unravel_index(int(np.argmax(heat)), heat.shape)
        records.append({"row_id": i + 1, "image": row["image"], "true": int(row["target"]),
                        "pred": pred, "prob_malignant": p_mal, "peak_x": int(px), "peak_y": int(py),
                        "rgb": rgb, "overlay": show_cam_on_image(rgb.astype(np.float32), heat, use_rgb=True)})

    n_pages = math.ceil(len(records) / args.per_page)
    for page in range(n_pages):
        chunk = records[page * args.per_page:(page + 1) * args.per_page]
        for annotated in ([False, True] if args.annotated else [False]):
            fig, axes = plt.subplots(len(chunk), 2, figsize=(6.4, 3.0 * len(chunk) + 0.6),
                                     squeeze=False, layout="constrained")
            for r, rec in enumerate(chunk):
                left, right = axes[r]
                left.imshow(rec["rgb"]); right.imshow(rec["overlay"])
                if args.mark_peak:
                    right.plot(rec["peak_x"], rec["peak_y"], marker="+", color="white", ms=14, mew=2)
                left.set_title(f"Row {rec['row_id']}: {rec['image']} (original)", fontsize=8)
                if annotated:
                    ok = rec["pred"] == rec["true"]
                    color = "green" if ok else "red"
                    right.set_title(f"true {config.CLASS_NAMES[rec['true']]} | pred "
                                    f"{config.CLASS_NAMES[rec['pred']]} (p malignant {rec['prob_malignant']:.2f})",
                                    fontsize=8, color=color)
                    for spine in right.spines.values():
                        spine.set_edgecolor(color); spine.set_linewidth(3)
                else:
                    right.set_title("Grad-CAM++ overlay", fontsize=8)
                for ax in (left, right):
                    ax.set_xticks([]); ax.set_yticks([])
            title = "AUTHOR REFERENCE (not for raters)" if annotated else RULE
            fig.suptitle(f"{title}\nPage {page + 1} of {n_pages}", fontsize=9)
            name = f"{'annotated' if annotated else 'rater_gallery'}_page_{page + 1:02d}.png"
            fig.savefig(os.path.join(out_dir, name), dpi=200, bbox_inches="tight")
            plt.close(fig)

    sheet = pd.DataFrame({"row_id": [r["row_id"] for r in records],
                          "image": [r["image"] for r in records], "miss": "", "notes": ""})
    sheet.to_csv(os.path.join(out_dir, "rater_sheet.csv"), index=False)
    details = pd.DataFrame([{k: r[k] for k in ("row_id", "image", "true", "pred", "prob_malignant",
                                               "peak_x", "peak_y")} for r in records])
    details.to_csv(os.path.join(out_dir, "author_cam_details.csv"), index=False)
    print(f"Wrote {n_pages} rater gallery page(s), rater_sheet.csv and author_cam_details.csv to {out_dir}")
    print("Give raters ONLY the rater_gallery pages and a copy of rater_sheet.csv.")


def progression(args):
    device = get_device(args.device)
    model = load_trained_model(args.model, device)
    x, rgb = prepare(os.path.join(args.images, args.image), build_transforms(train=False), device)
    pred, p_mal = predict(model, x)
    layers = [("Layer 3, final block (mid-level)", model.layer3[-1]),
              ("Layer 4, block 2 (high-level)", model.layer4[-2]),
              ("Layer 4, final block (used in the audit)", model.layer4[-1])]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb); axes[0].set_title("Original image")
    for ax, (name, layer) in zip(axes[1:], layers):
        ax.imshow(show_cam_on_image(rgb.astype(np.float32), cam_map(model, layer, x, pred), use_rgb=True))
        ax.set_title(name)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{args.image} | predicted {config.CLASS_NAMES[pred]} (p malignant {p_mal:.2f})", fontsize=14)
    fig.tight_layout()
    out_dir = ensure_dir(args.out)
    path = os.path.join(out_dir, f"progression_{stem(args.image)}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gallery", help="Blinded paired gallery and scoring sheet for raters")
    g.add_argument("--model", required=True)
    g.add_argument("--split-csv", required=True)
    g.add_argument("--images", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--per-page", type=int, default=10)
    g.add_argument("--shuffle-seed", type=int, default=7)
    g.add_argument("--no-shuffle", action="store_true")
    g.add_argument("--annotated", action="store_true",
                   help="Also write an author reference copy with labels and predictions")
    g.add_argument("--mark-peak", action="store_true",
                   help="Mark the activation maximum (the published scoring did not use a marker)")
    g.add_argument("--device", default=None)
    g.set_defaults(func=gallery)

    pr = sub.add_parser("progression", help="Layer-by-layer Grad-CAM++ for one image")
    pr.add_argument("--model", required=True)
    pr.add_argument("--images", required=True)
    pr.add_argument("--image", required=True, help="File name inside --images")
    pr.add_argument("--out", required=True)
    pr.add_argument("--device", default=None)
    pr.set_defaults(func=progression)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
