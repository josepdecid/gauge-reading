import torch


class IntersectionOverUnionLoss:
    def __init__(self, reduction='mean'):
        assert reduction in ['sum', 'mean']
        self.__reduction = reduction

    def __call__(self, a, b):
        intersection = IntersectionOverUnionLoss.__intersection__(a, b)

        # Area of boxes W * H
        area_a = torch.abs(a[:, 2] * a[:, 3])
        area_b = torch.abs(b[:, 2] * b[:, 3])

        union = area_a + area_b - intersection
        iou = -(intersection / union)

        if self.__reduction == 'mean':
            return iou.mean()
        else:
            return iou.sum()

    @staticmethod
    def __intersection__(a, b):
        min_x = torch.stack([a[:, 0], b[:, 0]])
        max_x = torch.stack([a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]])

        min_y = torch.stack([a[:, 1], b[:, 1]])
        max_y = torch.stack([a[:, 1] + a[:, 3], b[:, 1] + b[:, 3]])

        d_x = torch.clamp(torch.min(max_x, dim=0).values - torch.max(min_x, dim=0).values, min=0)
        d_y = torch.clamp(torch.min(max_y, dim=0).values - torch.max(min_y, dim=0).values, min=0)
        return d_x * d_y
