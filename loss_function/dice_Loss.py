import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, gt):

        intersection = (pred * gt).sum()

        gt_sum = gt.sum()
        pred_sum = pred.sum()

        loss = 1 - (2*intersection + 1)/(gt_sum + pred_sum + 1)

        return loss

