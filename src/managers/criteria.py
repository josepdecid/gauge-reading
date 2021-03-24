import torch


class IntersectionOverUnionLoss:
    """
    Intersection Over Union loss function

    Computes the Jaccard Similarity of two bounding boxes returning the intersection over the union.
    The resulting values is returned in negative as torch tries to minimize the loss function, and the target is
    to maximize the resulting value.

    It receives two pairs of N bounding boxes with shape N x 4 where N is the number of samples and the four elements
    in the second dimension correspond to the min_x, min_y, width, and height respectively.

    It is possible to specify between the reduction method for multiple batch elements (sum or mean).
    """

    def __init__(self, reduction='mean'):
        assert reduction in ['sum', 'mean']
        self.__reduction = reduction

    def __call__(self, a, b):
        intersection = IntersectionOverUnionLoss.__intersection__(a, b)

        # Area of boxes W * H
        area_a = torch.abs(a[:, 2] * a[:, 3])
        area_b = torch.abs(b[:, 2] * b[:, 3])

        # Union as the sum of both areas removing the intersected part
        union = area_a + area_b - intersection
        # Negative value aiming for a minimization
        iou = -(intersection / union)

        if self.__reduction == 'mean':
            return iou.mean()
        else:
            return iou.sum()

    @staticmethod
    def __intersection__(a, b):
        # Stack pair-wise coordinates
        min_x = torch.stack([a[:, 0], b[:, 0]])
        max_x = torch.stack([a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]])
        min_y = torch.stack([a[:, 1], b[:, 1]])
        max_y = torch.stack([a[:, 1] + a[:, 3], b[:, 1] + b[:, 3]])

        # Find the greater of the left-top part and the smaller of the right-bottom one
        d_x = torch.clamp(torch.min(max_x, dim=0).values - torch.max(min_x, dim=0).values, min=0)
        d_y = torch.clamp(torch.min(max_y, dim=0).values - torch.max(min_y, dim=0).values, min=0)
        return d_x * d_y
