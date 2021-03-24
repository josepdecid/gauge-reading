import json
import os
from glob import glob
from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GaugeDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.__root = root
        self.__transform = transform

        self.__data, self.__annotations = self.__make_dataset()

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img = Image.open(self.__data[idx])
        img = np.array(img)

        with open(self.__annotations[idx], mode='r') as json_file:
            annotations = json.load(json_file)

        if self.__transform is not None:
            img, annotations = self.__transform(img, annotations)

        return img, annotations

    def __make_dataset(self) -> Tuple[List[str], List[str]]:
        data_files = sorted(glob(os.path.join(self.__root, 'Data', '*')))
        annotation_files = sorted(glob(os.path.join(self.__root, 'Annotations', '*')))
        assert len(data_files) == len(annotation_files), 'There is a different number of data and annotation files.'

        data_samples = []
        annotations = []
        for data_file, annotation_file in zip(data_files, annotation_files):
            assert os.path.basename(data_file).split('.')[0] == os.path.basename(annotation_file).split('.')[0], \
                f'Data files do not match ({data_file} and {annotation_file})'

            data_samples.append(data_file)
            annotations.append(annotation_file)

        return data_samples, annotations


"""
# Alternative get item representation for built-in prediction models

num_objs = len(annotations['annotation'])
boxes = []
area = []
for i in range(num_objs):
    bbox = annotations['annotation'][i]['bbox']
    boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    area.append(annotations['annotation'][i]['area'])

target = {
    'boxes': torch.as_tensor(boxes, dtype=torch.float32),
    'area': torch.as_tensor(area, dtype=torch.float32),
    'labels': torch.ones((num_objs,), dtype=torch.int64),
    'is_crowd': torch.zeros((num_objs,), dtype=torch.int64),
    'image_id': annotations['image']['id'],
    'prediction': annotations['annotation'][0]['class_values'][0]
}
"""
