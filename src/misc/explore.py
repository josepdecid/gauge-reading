import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import patches

from transforms.random_background import RandomBackground
from transforms.random_gauge_offset import RandomGaugeOffset


def check_random_background_transformation():
    rb = RandomBackground('backgrounds')
    img = Image.open(os.path.join('datasets', 'training', 'Data', 'SC_1585054719260.png'))
    img = np.array(img)
    img, _ = rb(img, {})
    plt.imshow(img)
    plt.show()


def check_random_gauge_offset():
    ro = RandomGaugeOffset()
    img = Image.open(os.path.join('datasets', 'training', 'Data', 'SC_1585054719260.png'))
    img = np.array(img)
    annotations = json.load(open(os.path.join('datasets', 'training', 'Annotations', 'SC_1585054719260.json')))

    img, annotations = ro(img, annotations)

    f, ax = plt.subplots()
    plt.imshow(img)
    bbox = annotations['annotation'][0]['bbox']
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


if __name__ == '__main__':
    # check_random_background_transformation()
    check_random_gauge_offset()
