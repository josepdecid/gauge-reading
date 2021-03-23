import cv2
import numpy as np


class ResizeImage(object):
    def __init__(self, resize_factor: float):
        self.__resize_factor = resize_factor

    def __call__(self, img: np.array, annotations: dict):
        new_x = int(img.shape[1] * self.__resize_factor)
        new_y = int(img.shape[0] * self.__resize_factor)
        img = cv2.resize(img, dsize=(new_x, new_y), interpolation=cv2.INTER_CUBIC)

        b_x, b_y, width, height = annotations['annotation'][0]['bbox']
        annotations['annotation'][0]['bbox'] = [
            int(b_x * self.__resize_factor),
            int(b_y * self.__resize_factor),
            int(width * self.__resize_factor),
            int(height * self.__resize_factor)
        ]

        return img, annotations
