import argparse

from managers.trainer import Trainer
from models.model_helpers import resnet18_for_bbox, resnet18_for_regression

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN to solve the Gauge Reading')
    parser.add_argument('data_path', type=str, help='Dataset folder path')
    parser.add_argument('bbox_checkpoint', type=str, default=None, help='Dataset folder root path')
    parser.add_argument('value_checkpoint', type=str, default=None, help='Dataset folder root path')
    args = parser.parse_args()

    bbox_model = resnet18_for_bbox(pretrained=True)
    value_model = resnet18_for_regression(pretrained=True)

    trainer = Trainer(
        bbox_model=bbox_model,
        value_model=value_model,
        root_path=args.data_path,
        batch_size=1,
        bbox_checkpoint=args.bbox_checkpoint,
        value_checkpoint=args.value_checkpoint
    )

    trainer.predict()
