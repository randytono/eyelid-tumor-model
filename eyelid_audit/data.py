"""Dataset and image transforms."""

import os

import numpy as np
import pandas as pd
from PIL import Image

from .config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def build_transforms(train):
    """Albumentations pipelines used in the paper.

    Training: bicubic resize, rotation (+/-20 deg), horizontal flip,
    brightness/contrast (+/-0.3), coarse dropout (1-8 holes, 8-32 px), each with
    p = 0.5, then ImageNet normalization. Validation and test: resize and
    normalization only.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    resize = A.Resize(IMG_SIZE, IMG_SIZE, interpolation=3)  # 3 = bicubic
    normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if not train:
        return A.Compose([resize, normalize, ToTensorV2()])
    return A.Compose([
        resize,
        A.Rotate(limit=20, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        # Albumentations >= 2.0 argument names
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32),
                        hole_width_range=(8, 32), p=0.5),
        normalize,
        ToTensorV2(),
    ])


def load_rgb(path):
    """Load an image as an RGB uint8 array. Fails loudly if the file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return np.array(Image.open(path).convert("RGB"))


def denormalize(tensor_chw):
    """Undo ImageNet normalization for display. Returns an HxWx3 float array in [0, 1]."""
    img = tensor_chw.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.array(IMAGENET_STD) * img + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)


class TumorDataset:
    """Reads a split CSV with columns `image` and `target` (0 benign, 1 malignant).

    Set return_name=True to also return the image file name.
    """

    def __init__(self, csv_file, img_dir, transform=None, return_name=False):
        self.df = pd.read_csv(csv_file)
        missing = {"image", "target"} - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_file} is missing columns: {sorted(missing)}")
        self.img_dir = img_dir
        self.transform = transform
        self.return_name = return_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import torch

        row = self.df.iloc[idx]
        image = load_rgb(os.path.join(self.img_dir, row["image"]))
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        label = torch.tensor(int(row["target"]), dtype=torch.long)
        if self.return_name:
            return image, label, row["image"]
        return image, label
