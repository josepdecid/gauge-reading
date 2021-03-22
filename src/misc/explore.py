import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from transforms.random_background import RandomBackground


def check_random_background_transformation():
    rb = RandomBackground('backgrounds')
    img = Image.open(os.path.join('datasets', 'training', 'Data', 'SC_1585054719260.png'))
    img = np.array(img)
    img, _ = rb(img, {})
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    check_random_background_transformation()

    """
    for idx, (img, b_boxes, target) in enumerate(loader):
        img = img[0].permute(1, 2, 0).numpy()

        f, ax = plt.subplots()
        ax.imshow(img)

        bbox = list(map(lambda x: x.item(), b_boxes))
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.show()

        if idx == 9:
            break
    """
