import os
from glob import glob
from logging import warning

import cv2
import numpy as np
from PIL import Image


class RandomBackground(object):
    """

    """

    def __init__(self, folder_path: str, min_cropping_percentage: int = 0.25):
        background_files = glob(os.path.join(folder_path, '*'))
        assert os.path.exists(folder_path), 'The specified background folder does not exist.'
        assert len(background_files), 'There are no available backgrounds in the specified folder.'
        self.__backgrounds = list(filter(lambda x: x.split('.')[-1] in ['png', 'jpg'], background_files))

        assert 0 <= min_cropping_percentage <= 1, 'The cropping percentage must be in range [0, 1].'
        self.__min_cropping_percentage = min_cropping_percentage

    def __call__(self, img, annotations):
        if img.shape[2] == 3:
            warning('Image does not have an alpha channel, skipping this step.')
            return img[:, :, :3], annotations

        # Select a random background
        random_idx = np.random.randint(0, len(self.__backgrounds))
        random_bg = np.array(Image.open(self.__backgrounds[random_idx]))

        # Crop the background randomly
        cropping_percentage = np.random.uniform(low=self.__min_cropping_percentage, high=1.0)
        cropped_width = int(random_bg.shape[0] * cropping_percentage)
        cropped_height = int(random_bg.shape[1] * cropping_percentage)
        min_x = np.random.randint(0, random_bg.shape[0] - cropped_width)
        min_y = np.random.randint(0, random_bg.shape[1] - cropped_height)
        random_bg = random_bg[min_x:min_x + cropped_width, min_y:min_y + cropped_height, :]

        # Resize image to the same size of the original image
        random_bg = cv2.resize(random_bg, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Merge gauge and background according to the alpha channels of the original image
        img = np.where((img[:, :, 3] == 255)[..., None], img[:, :, :3], random_bg)

        return img, annotations
