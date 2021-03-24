import cv2
import numpy as np


class RandomGaugeOffset(object):
    """
    It extracts the Gauge from an image with an alpha channel, separating the foreground and the background.
    The gauge itself is cropped by a random factor between `min_cropping_factor` and `max_cropping_factor`.
    Then, it applies an offset, moving it to a new random position, also recomputing the bounding box coordinates.
    """

    def __init__(self, min_cropping_factor: int = 0.25, max_cropping_factor: int = 1.5):
        self.__min_cropping_factor = min_cropping_factor
        self.__max_cropping_factor = max_cropping_factor

    def __call__(self, img: np.array, annotations: dict):
        gauge_img = np.where(img[:, :, 3] == 255)
        min_x, max_x = np.min(gauge_img[0]), np.max(gauge_img[0])
        min_y, max_y = np.min(gauge_img[1]), np.max(gauge_img[1])
        gauge_img = img[min_x:max_x, min_y:max_y].copy()

        # Resize the gauge
        cropping_percentage = np.random.uniform(low=self.__min_cropping_factor, high=self.__max_cropping_factor)
        cropped_x = min(int(gauge_img.shape[0] * cropping_percentage), img.shape[0] - 1)
        cropped_y = min(int(gauge_img.shape[1] * cropping_percentage), img.shape[1] - 1)
        gauge_img = cv2.resize(gauge_img, dsize=(cropped_y, cropped_x), interpolation=cv2.INTER_CUBIC)

        # Calculate new top-left coordinates
        new_x = np.random.randint(0, img.shape[0] - gauge_img.shape[0])
        new_y = np.random.randint(0, img.shape[1] - gauge_img.shape[1])

        img = np.zeros_like(img)
        img[new_x:new_x + gauge_img.shape[0], new_y:new_y + gauge_img.shape[1], :] = gauge_img

        # Modify accordingly the bounding boxes
        b_x, b_y, width, height = annotations['annotation'][0]['bbox']
        annotations['annotation'][0]['bbox'] = [
            new_y + (b_y - min_x) * cropping_percentage,
            new_x + (b_x - min_y) * cropping_percentage,
            width * cropping_percentage,
            height * cropping_percentage
        ]

        return img, annotations
