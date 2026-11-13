"""Grad-CAM: where in the photo the network found what pushed its score up.

For one photo, the last convolutional block gives a grid of feature maps (16 x 10 cells for a
512 x 320 input). Grad-CAM weighs each map by how much the score grows with it and adds them
up: cells with a high value are the regions that argued for "forged". It explains the score of
one model on one photo; it does not say the region is forged, only that the model leaned on it.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

HEAT_COLOR = (235, 104, 52)  # one warm hue: the photo stays visible where the model did not look


def grad_cam(model: nn.Module, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Scores and heat maps for a batch of normalized photos (N x 3 x H x W).

    Returns (scores N, maps N x h x w). Each map is scaled to [0, 1] by its own maximum, so it
    shows where the model looked, not how strongly; a photo with nothing in favour of "forged"
    gets a map of zeros. The model must be one from net.build (features, avgpool, classifier)."""
    model.eval()
    # Only the gradient of the score with respect to the last feature maps is needed, so the
    # convolutional base runs without keeping gradients: it is most of the memory and the time.
    with torch.no_grad():
        features = model.features(x)
    features = features.detach().requires_grad_(True)
    with torch.enable_grad():
        logits = model.classifier(torch.flatten(model.avgpool(features), 1)).squeeze(1)
        (gradient,) = torch.autograd.grad(logits.sum(), features)
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    maps = F.relu((weights * features).sum(dim=1)).detach()
    peak = maps.flatten(1).max(dim=1).values.clamp_min(1e-12).view(-1, 1, 1)
    maps = torch.where(maps.flatten(1).max(dim=1).values.view(-1, 1, 1) > 0, maps / peak, torch.zeros_like(maps))
    return torch.sigmoid(logits.detach()).float().cpu().numpy(), maps.float().cpu().numpy()


def overlay(photo: np.ndarray, heat: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """The photo (uint8 H x W x 3) with the heat map painted over it. The photo is turned to
    grey so the only colour is the heat; the heat map is enlarged smoothly to the photo size."""
    height, width = photo.shape[:2]
    grey = np.asarray(Image.fromarray(photo).convert("L"), dtype=np.float32)[..., None].repeat(3, axis=2)
    large = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize((width, height), Image.BICUBIC),
                       dtype=np.float32)[..., None] / 255
    alpha = np.clip(large, 0, 1) * strength
    return (grey * (1 - alpha) + np.array(HEAT_COLOR, dtype=np.float32) * alpha).round().astype(np.uint8)


def border_share(heat: np.ndarray, margin: float = 0.125) -> float:
    """Share of the heat that falls in the outer frame of the photo (`margin` of each side). With
    heat spread evenly it equals the share of the area of that frame; a model that looks at the
    background instead of the card gives more."""
    total = float(heat.sum())
    if total <= 0:
        return float("nan")
    rows, cols = heat.shape
    top, left = max(1, round(rows * margin)), max(1, round(cols * margin))
    inner = float(heat[top:rows - top, left:cols - left].sum())
    return 1 - inner / total
