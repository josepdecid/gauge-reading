import numpy as np
import matplotlib.pyplot as plt


class RandomGaugeOffset(object):
    def __call__(self, img, annotations):
        f, ax = plt.subplots(1, 2)
        img = np.array(img)
        ax[0].imshow(img)
        bbox = annotations['boxes'][0].numpy().astype(int)

        gauge_img = img[bbox[0] - 20:bbox[2] + 20, bbox[1] - 20:bbox[3] + 20, :].copy()
        new_x = np.random.randint(0, img.shape[0] - gauge_img.shape[0])
        new_y = np.random.randint(0, img.shape[1] - gauge_img.shape[1])

        img = np.zeros_like(img)
        img[new_x, new_y, :] = gauge_img
        ax[1].imshow(img)
        plt.show()

        return img, annotations
