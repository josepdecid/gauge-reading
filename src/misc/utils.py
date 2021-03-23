import io
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches


def display_predictions(images, predictions, targets: Optional, show=True):
    results = []

    if targets is not None:
        targets

    for idx, (image, pred) in enumerate(zip(images, predictions)):
        image = image.permute(1, 2, 0).cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()

        f, ax = plt.subplots()
        ax.imshow(image)

        rect = patches.Rectangle((pred[0], pred[1]), pred[2], pred[3],
                                 linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        if targets is not None:
            target = targets[idx].cpu().detach().numpy()
            rect = patches.Rectangle((target[0], target[1]), target[2], target[3],
                                     linewidth=1, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

        if show:
            plt.show()

        # result = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        result = get_img_from_fig(f)
        result = (torch.from_numpy(result) / 255.0).permute(2, 0, 1)
        results.append(result)

    return torch.stack(results)


def get_img_from_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(128, 128))
    plt.close()

    return img
