import matplotlib.pyplot as plt
from matplotlib import patches


def collate_fn(batch):
    return tuple(zip(*batch))


def display_predictions(images, predictions):
    for image, prediction in zip(images, predictions):
        image = image.permute(1, 2, 0).cpu().numpy()

        f, ax = plt.subplots()
        ax.imshow(image)

        x_min, y_min, x_max, y_max = prediction['boxes'][0].tolist()
        width = x_max - x_min
        height = y_max - y_min

        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.show()
