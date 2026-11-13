"""Training of one model and scoring with it.

One call to `fit` trains a model on the training people, watching a separate validation set:
after every epoch it measures the PR-AUC there, keeps the best weights and stops when it has
not improved for `patience` epochs. The test fold is never seen here.

The augmentation is mild on purpose (brightness, contrast, a few degrees of rotation, a small
shift and zoom) and never flips or cuts the card: that would destroy what gives a forgery away.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..evaluation.metrics import average_precision
from .data import IMAGENET_MEAN, IMAGENET_STD
from .net import keep_frozen_in_eval, set_trainable, trainable_parameters


@dataclass
class TrainConfig:
    head_epochs: int = 3          # stage 1: only the head
    tune_epochs: int = 20         # stage 2: at most this many, early stopping decides
    trainable_blocks: int = 6     # blocks of the base released in stage 2
    head_lr: float = 1e-3
    tune_lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    patience: int = 6
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


def to_input(batch: np.ndarray, device) -> torch.Tensor:
    """uint8 N x H x W x 3 -> float N x 3 x H x W in [0, 1] on the device."""
    return torch.from_numpy(np.ascontiguousarray(batch)).to(device).permute(0, 3, 1, 2).float().div_(255)


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Mild random changes, different for each photo of the batch (x in [0, 1])."""
    n, _, height, width = x.shape

    def uniform(low: float, high: float) -> torch.Tensor:
        return torch.rand(n, device=x.device, generator=generator) * (high - low) + low

    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * uniform(0.85, 1.15).view(n, 1, 1, 1) + mean) * uniform(0.85, 1.15).view(n, 1, 1, 1)
    angle = uniform(-4, 4) * math.pi / 180
    zoom = uniform(0.95, 1.05)
    cos, sin = torch.cos(angle) / zoom, torch.sin(angle) / zoom
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin * height / width, uniform(-0.03, 0.03)
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin * width / height, cos, uniform(-0.03, 0.03)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False).clamp(0, 1)


@torch.no_grad()
def predict(model: nn.Module, photos: np.ndarray, indices: np.ndarray, device, batch_size: int = 64) -> np.ndarray:
    """Probability of forgery for the photos at `indices` (no augmentation)."""
    model.eval()
    scores = []
    for start in range(0, len(indices), batch_size):
        batch = np.sort(indices[start:start + batch_size])  # sorted: sequential reads from the memory map
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(normalize(to_input(photos[batch], device))).squeeze(1)
        scores.append(dict(zip(batch.tolist(), torch.sigmoid(logits.float()).cpu().numpy().tolist())))
    merged = {index: score for part in scores for index, score in part.items()}
    return np.array([merged[int(i)] for i in indices])


def fit(model: nn.Module, photos: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray | None,
        cfg: TrainConfig, device, log=print) -> list[dict]:
    """Train `model` in place and leave it with the weights of its best validation epoch.
    With `val_idx` None there is nothing to watch (the final model, trained on everyone): it
    trains exactly the epochs of the configuration and keeps the weights of the last one.
    `photos` is the uint8 array of every person, `labels` 1 = forged, 0 = genuine. Returns the
    history: one row per epoch, with the loss and the accuracy (threshold 0.5) on training and
    on validation, and the validation PR-AUC, which is the one that decides the best epoch. The
    validation loss weighs the positives like the training loss, so the two curves compare."""
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    shuffler = np.random.default_rng(cfg.seed)
    y_train = labels[train_idx]
    positives, negatives = int(y_train.sum()), int((y_train == 0).sum())
    pos_weight = negatives / max(1, positives)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    y_val = None if val_idx is None else labels[val_idx].astype(bool)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history, best, best_state, since_best = [], -1.0, None, 0

    stages = [("head", cfg.head_epochs, 0, cfg.head_lr), ("tune", cfg.tune_epochs, cfg.trainable_blocks, cfg.tune_lr)]
    for stage, epochs, blocks, lr in stages:
        if epochs <= 0:
            continue
        set_trainable(model, blocks)
        optimizer = torch.optim.AdamW(trainable_parameters(model), lr=lr, weight_decay=cfg.weight_decay)
        since_best = 0  # the patience of the second stage does not count the epochs of the first
        for epoch in range(1, epochs + 1):
            model.train()
            keep_frozen_in_eval(model)
            order, losses, hits, seen = shuffler.permutation(train_idx), [], 0, 0
            for start in range(0, len(order), cfg.batch_size):
                batch = np.sort(order[start:start + cfg.batch_size])
                if len(batch) < 2:  # batch normalization cannot train on a single photo
                    continue
                x = normalize(augment(to_input(photos[batch], device), generator))
                y = torch.from_numpy(labels[batch]).float().to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    logits = model(x).squeeze(1).float()
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
                hits += int(((logits.detach() > 0) == (y > 0.5)).sum())
                seen += len(batch)
            if val_idx is None:
                row = {"stage": stage, "epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                       "train_accuracy": round(hits / max(1, seen), 4)}
                history.append(row)
                log(f"      {stage} {epoch:2d}  loss {row['train_loss']:.4f}  accuracy {row['train_accuracy']:.4f}")
                continue
            scores = predict(model, photos, val_idx, device)
            val_pr_auc = average_precision(labels[val_idx], scores)
            clipped = np.clip(scores, 1e-7, 1 - 1e-7)
            val_loss = -np.mean(np.where(y_val, pos_weight * np.log(clipped), np.log(1 - clipped)))
            row = {"stage": stage, "epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                   "train_accuracy": round(hits / max(1, seen), 4), "val_loss": round(float(val_loss), 4),
                   "val_accuracy": round(float(((scores > 0.5) == y_val).mean()), 4),
                   "val_pr_auc": round(val_pr_auc, 4)}
            history.append(row)
            log(f"      {stage} {epoch:2d}  loss {row['train_loss']:.4f}  validation: loss {row['val_loss']:.4f}  "
                f"accuracy {row['val_accuracy']:.4f}  PR-AUC {row['val_pr_auc']:.4f}" + ("  *" if val_pr_auc > best else ""))
            if val_pr_auc > best:
                best, since_best = val_pr_auc, 0
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            else:
                since_best += 1
                if stage == "tune" and since_best >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
