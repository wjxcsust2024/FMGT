import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):  #score(Dicss) + loss1(BCE)
        p = logits.view(-1, 1)
        t = targets.view(-1, 1)
        loss1 = F.binary_cross_entropy_with_logits(p, t, reduction='mean')

        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num

        return score + loss1  #（0.2、0.8）#只能在这里调整参数