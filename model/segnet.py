import torch.nn as nn
import backbones
import detector
from loss_function.dice_Loss import DiceLoss

class SegNet(nn.Module):

    def __init__(self, device, **kwargs):
        super(SegNet, self).__init__()
        self.backbone = backbones.resnet18(**kwargs)
        self.SEGdetector = detector.SEGDetecor()
        self.device = device
        self.criterion = DiceLoss()
        self.to(self.device)

    def forward(self, batch):
        features = self.backbone(batch['image'].cuda())
        pred = self.SEGdetector(features)

        if 'gt' in batch:
            gt = batch['gt'].cuda()
            loss = self.criterion(pred, gt)
            return loss, pred

        return pred

# if __name__=='__main__':
#     model = SegNet()
#     print(model)