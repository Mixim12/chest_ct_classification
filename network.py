import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights


class Network(nn.Module):
    def __init__(self, num_classes=4):
        super(Network, self).__init__()
        self.model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)
