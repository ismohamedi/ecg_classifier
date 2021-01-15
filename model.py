import torch.nn as nn
from torchvision import models


def ResNet34(num_classes=8):
    model = models.resnet34()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
