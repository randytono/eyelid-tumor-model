"""Step 2. Fine-tune ResNet-50 and keep the best-validation-accuracy checkpoint.

    python scripts/02_train.py --splits data/splits --images data/images --out models

The published model was trained for 20 epochs without a fixed random seed, so
retraining gives similar but not identical weights. Pass --seed for runs you
want to repeat exactly on the same hardware.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from eyelid_audit import config
from eyelid_audit.data import TumorDataset, build_transforms
from eyelid_audit.model import adaptive_focal_loss, build_model
from eyelid_audit.utils import ensure_dir, get_device, set_seed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", required=True, help="Folder with train.csv and val.csv")
    ap.add_argument("--images", required=True, help="Folder with the image files")
    ap.add_argument("--out", required=True, help="Folder for best_model.pth and training_log.csv")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--seed", type=int, default=None, help="Optional random seed (published run: none)")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="Start from random weights (offline testing only)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dir(args.out)
    print(f"Device: {device} | seed: {args.seed}")

    train_ds = TumorDataset(os.path.join(args.splits, "train.csv"), args.images, build_transforms(train=True))
    val_ds = TumorDataset(os.path.join(args.splits, "val.csv"), args.images, build_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = adaptive_focal_loss(config.FOCAL_GAMMA)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_acc, best_epoch = 0.0, None
    log_path = os.path.join(args.out, "training_log.csv")
    with open(log_path, "w", newline="") as log_file:
        log = csv.writer(log_file)
        log.writerow(["epoch", "train_loss", "val_accuracy", "saved"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
                running += loss.item()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    preds = torch.argmax(model(imgs), dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            train_loss = running / max(len(train_loader), 1)
            val_acc = correct / max(total, 1)
            saved = val_acc > best_val_acc
            if saved:
                best_val_acc, best_epoch = val_acc, epoch
                torch.save(model.state_dict(), os.path.join(args.out, "best_model.pth"))
            log.writerow([epoch, f"{train_loss:.6f}", f"{val_acc:.6f}", int(saved)])
            log_file.flush()
            print(f"Epoch {epoch:>3} | loss {train_loss:.4f} | val acc {val_acc:.4f}" + ("  <- saved" if saved else ""))

    if best_epoch is None:
        sys.exit("Validation accuracy never rose above 0; no checkpoint was saved.")
    print(f"Best validation accuracy {best_val_acc:.4f} at epoch {best_epoch}. Log: {log_path}")


if __name__ == "__main__":
    main()
