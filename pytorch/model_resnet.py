"""Pretrained Models for finetuning
"""
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

class PretrainedResNet50(nn.Module):
    """Pretrained ResNet50 model"""

    def __init__(
        self,
        num_classes: int = 10,
        lr: float = 1e-4,
        dropout_rate: float = 0.3,
        pretrained_weights=True,
        freeze: Union[bool, int] = False,
    ):
        """Pretrained ResNet50

        Args:
            num_classes (int, optional): Number of Classes. Defaults to 10.

            lr (float, optional): Learning Rate. Defaults to 1e-4.

            dropout_rate (float, optional): Dropout Rate. Defaults to 0.3.

            pretrained_weights (bool, optional): Whether to load pretrained weights. Defaults to False.

            freeze (Union[bool, int], optional): bool or int. If True, freeze
            weights for all layers under ResNet model. If False, do not freeze
            weights for any layers. If given int value, freeze weights of the
            first (bottom) number of layers. Defaults to False.
        """
        super(PretrainedResNet50, self).__init__(num_classes, lr, dropout_rate)
        if pretrained_weights:
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.model = resnet50()

        self.model.fc = nn.Idenity()

        self.fc1 = nn.Linear(self.featveclen, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

    def forward(self, inputs):
        """Forward Propagation step

        Args:
            inputs: torch.Tensor. Dimension = (batch_size, n_channels, 28, 28)

        Returns:
            torch.Tensor. Dimension = (batch_size, num_classes)
        """
        x = self.model(inputs)
        x = F.selu(self.dropout(self.fc1(x)))
        x = F.selu(self.dropout(self.fc2(x)))
        out = self.fc3(x)
        return out

if __name__ == "__main__":
    resnet = PretrainedResNet50()
    sample_img = torch.randn(5, 3, 224, 224)
    sample_outs = resnet(sample_img)
    print(sample_outs.size())