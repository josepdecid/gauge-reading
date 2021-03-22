from matplotlib import patches
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
import matplotlib.pyplot as plt

from managers.dataset import Dataset

if __name__ == '__main__':
    dataset = Dataset('datasets\\training', Compose([ToTensor()]))
    loader = DataLoader(dataset)

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
