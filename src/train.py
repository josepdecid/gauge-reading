import argparse

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from managers.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN to solve the Gauge Reading')
    parser.add_argument('data_root_path', type=str, help='Dataset folder root path')
    parser.add_argument('--epochs', type=int, default=100, help='#Epochs')
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--seed', type=int, help='Fixed seed to ensure reproducibility')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    """
    backbone = mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    """

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    trainer = Trainer(
        model=model,
        root_path=args.data_root_path,
        batch_size=2
    )

    trainer.train(epochs=10)
