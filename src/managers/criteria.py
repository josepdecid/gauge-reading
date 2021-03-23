import torch
from torch.nn import MSELoss


class IntersectionOverUnionLoss:
    def __init__(self, reduction='mean'):
        assert reduction in ['sum', 'mean']
        self.__reduction = reduction

    def __call__(self, box_x, box_y):
        area_x = box_x[:, 2] * box_x[:, 3]
        area_y = box_y[:, 2] * box_y[:, 3]

        intersection = IntersectionOverUnionLoss.__intersection__(box_x, box_y)

        union = area_x + area_y - intersection
        iou = - (intersection / union)

        if self.__reduction == 'sum':
            return iou.sum()
        else:
            return iou.mean()

    @staticmethod
    def __intersection__(x, y):
        d_x = torch.min((x[:, 0] + x[:, 2]) - (y[:, 0] + y[:, 2])) - torch.max(x[:, 0] - y[:, 0])
        d_y = torch.min((x[:, 1] + x[:, 3]) - (y[:, 1] + y[:, 3])) - torch.max(x[:, 1] - y[:, 1])
        return torch.clamp(d_x * d_y, min=0)
