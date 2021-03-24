import numpy as np
import torch
import torchvision


##
# Generic torchvision-like transform functions adapted to work with extra annotations parameter
##

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


class Normalize(object):
    def __init__(self, mean, std):
        self.__normalize = torchvision.transforms.Normalize(mean, std)

    def __call__(self, img: torch.FloatTensor, annotations: dict):
        img = self.__normalize(img)
        annotations['bbox'] = (np.array(annotations['bbox']) / torch.tensor(img.size()[1:]).repeat(2)).tolist()
        return img, annotations


class UnNormalize(object):
    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, img: torch.FloatTensor, predictions: torch.FloatTensor):
        for t, m, s in zip(img, self.__mean, self.__std):
            t.mul_(s).add_(m)

        predictions = predictions.cpu().detach().numpy()
        predictions = (predictions * np.tile(img.size()[1:], 2)).astype(int).tolist()

        return img, predictions
