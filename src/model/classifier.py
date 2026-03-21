"""Cat vs Dog classifier using MobileNetV2 with transfer learning."""

import torch
import torch.nn as nn
from torchvision import models


class CatDogClassifier(nn.Module):
    """MobileNetV2-based binary classifier.

    Uses a pretrained MobileNetV2 backbone with a custom classification head.
    The backbone can be frozen to train only the head (faster, less data needed)
    or unfrozen for full fine-tuning (better accuracy, needs lower learning rate).
    """

    def __init__(
        self,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Optionally unfreeze the last N feature layers for fine-tuning
            if unfreeze_last_n > 0:
                features = list(self.backbone.features)
                for layer in features[-unfreeze_last_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
