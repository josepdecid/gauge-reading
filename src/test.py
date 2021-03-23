import argparse

import torch
from torch import nn
from torchvision import models

from managers.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN to solve the Gauge Reading')
    parser.add_argument('data_path', type=str, help='Dataset folder path')
    parser.add_argument('checkpoint_path', type=str, default=None, help='Dataset folder root path')
    args = parser.parse_args()

    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    """

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 4)

    trainer = Trainer(
        model=model,
        root_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=1
    )

    trainer.predict()
