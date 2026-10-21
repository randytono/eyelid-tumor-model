"""ResNet-50 classifier and the focal loss used in training."""


def build_model(pretrained=True, freeze_backbone=True):
    """ImageNet-pretrained ResNet-50 with a custom two-class head.

    Only `layer4` and the new head are trainable when freeze_backbone=True.
    Head: Linear(2048, 1024) -> BatchNorm1d -> ReLU -> Dropout(0.5) -> Linear(1024, 2).
    """
    import torch.nn as nn
    from torchvision import models

    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 2),
    )
    return model


def load_trained_model(weights_path, device):
    """Rebuild the architecture and load trained weights in eval mode."""
    import torch

    model = build_model(pretrained=False, freeze_backbone=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def adaptive_focal_loss(gamma=2.0):
    """Focal loss with a batch-adaptive weight alpha = 1 - mean(batch labels).

    Note: alpha is a single scalar applied to the whole batch loss, not the
    per-class alpha of the original focal loss (Lin et al., 2017).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class AdaptiveFocalLoss(nn.Module):
        def __init__(self, gamma):
            super().__init__()
            self.gamma = gamma

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")
            pt = torch.exp(-ce_loss)
            alpha = 1.0 - (targets.float().mean() + 1e-6)
            return (alpha * (1 - pt) ** self.gamma * ce_loss).mean()

    return AdaptiveFocalLoss(gamma)
