import os
from glob import glob
from logging import warning

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class RandomBackground(object):
    def __init__(self, folder_path: str):
        background_files = glob(os.path.join(folder_path, '*'))
        self.__backgrounds = list(filter(lambda x: x.split('.')[-1] in ['png', 'jpg'], background_files))

    def __call__(self, img, annotations):
        if img.shape[2] == 3:
            warning('Image does not have an alpha channel, skipping this step.')
            return img, annotations

        # Select a random background
        random_idx = np.random.randint(0, len(self.__backgrounds))
        random_bg = np.array(Image.open(self.__backgrounds[random_idx]))

        # Crop the background randomly
        cropping_percentage = np.random.random()
        cropped_width = int(random_bg.shape[0] * cropping_percentage)
        cropped_height = int(random_bg.shape[1] * cropping_percentage)
        min_x = np.random.randint(0, random_bg.shape[0] - cropped_width)
        min_y = np.random.randint(0, random_bg.shape[1] - cropped_height)
        random_bg = random_bg[min_x:min_x + cropped_width, min_y:min_y + cropped_height, :]

        # Resize image to the same size of the original image
        random_bg = cv2.resize(random_bg, dsize=(img.shape[0], img.shape[1]), interpolation=cv2.INTER_CUBIC)

        # Merge gauge and background according to the alpha channels of the original image
        img = np.where((img[:, :, 3] == 255)[..., None], img[:, :, :3], random_bg)

        return img, annotations


if __name__ == '__main__':
    def main():
        rb = RandomBackground('backgrounds')

        img = Image.open('C:\\Users\\jdeci\\Projects\\GaugeReading\\datasets\\training\\Data\\SC_1585054719260.png')
        img = np.array(img)
        img, _ = rb(img, {})
        plt.imshow(img)
        plt.show()


    main()
