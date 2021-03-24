import argparse

import numpy as np
import torch

from managers.trainer import Trainer
from models.model_helpers import resnet18_for_bbox, resnet18_for_regression

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN to solve the Gauge Reading')
    parser.add_argument('data_root_path', type=str, help='Dataset folder root path')
    parser.add_argument('--bbox_checkpoint', type=str, default=None, help='Checkpoint for the Bbox model')
    parser.add_argument('--value_checkpoint', type=str, default=None, help='Checkpoint for the value model')
    parser.add_argument('--epochs', type=int, default=100, help='#Epochs')
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--seed', type=int, help='Fixed seed to ensure reproducibility')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    bbox_model = resnet18_for_bbox(pretrained=True)
    value_model = resnet18_for_regression(pretrained=True)

    trainer = Trainer(
        bbox_model=bbox_model,
        value_model=value_model,
        root_path=args.data_root_path,
        batch_size=args.bs,
        bbox_checkpoint=args.bbox_checkpoint,
        value_checkpoint=args.value_checkpoint
    )

    trainer.train(epochs=args.epochs)

"""
# Other possible models to try

backbone = mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
"""
