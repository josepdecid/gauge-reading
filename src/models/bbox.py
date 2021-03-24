from torch import nn
from torchvision import models


def resnet18_for_bbox(pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Sequential(
        nn.Linear(512, 4),
        nn.Sigmoid()
    )

    return model
