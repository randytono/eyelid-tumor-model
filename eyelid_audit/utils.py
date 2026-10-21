"""Small helpers shared by the scripts."""

import os
import random

import numpy as np

from .config import FEMALE_LABELS, MALE_LABELS


def get_device(preferred=None):
    """Return a torch device: the requested one, else CUDA, else Apple MPS, else CPU."""
    import torch

    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    """Seed Python, NumPy and PyTorch. Pass None to leave runs unseeded."""
    if seed is None:
        return
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_sex(value):
    """Map free-text sex labels to 'male', 'female' or 'unknown'."""
    v = str(value).strip().lower()
    if v in MALE_LABELS:
        return "male"
    if v in FEMALE_LABELS:
        return "female"
    return "unknown"


def stem(filename):
    """File name without extension, e.g. '0002.jpg' -> '0002'."""
    return os.path.splitext(str(filename).strip())[0]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
