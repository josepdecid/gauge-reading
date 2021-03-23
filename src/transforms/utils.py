import numpy as np
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img: np.array, annotations: dict):
        for t in self.transforms:
            img, annotations = t(img, annotations)
        return img, annotations


class ToTensor(object):
    def __call__(self, img: np.array, annotations: dict):
        img = torch.from_numpy(img[:, :, :3] / 255.0).type(torch.FloatTensor)
        img = img.permute(2, 0, 1)

        annotations = {
            'bbox': annotations['annotation'][0]['bbox'],
            'target': annotations['annotation'][0]['class_values'][0]
        }

        return img, annotations
