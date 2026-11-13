"""The network: an ImageNet model from torchvision with a new one-output head.

Transfer learning in two stages (same recipe as the course workshop, here in PyTorch):
  head   the pretrained base is frozen and only the new head is trained;
  tune   the last blocks of the base are released and trained with a low learning rate.
"""
import torch
from torch import nn
from torchvision import models

ARCHITECTURES = ("mobilenet_v3_large", "efficientnet_b0")


def build(arch: str = "mobilenet_v3_large", pretrained: bool = True, dropout: float | None = None) -> nn.Module:
    """Model with a single logit (forged vs genuine). `pretrained` downloads the ImageNet
    weights the first time (public file, nothing is uploaded). `dropout` replaces the rate of
    the dropout layer before the output (both networks come with 0.2)."""
    if arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"unknown architecture '{arch}' (choose one of {ARCHITECTURES})")
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, 1)
    if dropout is not None:
        for layer in model.classifier:
            if isinstance(layer, nn.Dropout):
                layer.p = dropout
    return model


def set_trainable(model: nn.Module, blocks: int) -> None:
    """Freeze the base except its last `blocks` blocks (0 = only the head is trained). Frozen
    blocks also keep their batch-norm statistics (see keep_frozen_in_eval)."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if blocks > 0:
        for block in list(model.features)[-blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def keep_frozen_in_eval(model: nn.Module) -> None:
    """Call after model.train(): blocks with no trainable parameter stay in eval mode, so their
    batch-norm statistics do not drift toward the small training set."""
    for block in model.features:
        if not any(parameter.requires_grad for parameter in block.parameters()):
            block.eval()


def trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
