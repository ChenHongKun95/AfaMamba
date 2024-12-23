import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss
# from .detail_loss import DetailAggregateLoss
from torch.autograd import Variable


def get_boundary(x):
    laplacian_kernel_target = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).cuda(device=x.device)
    x = x.unsqueeze(1).float()
    x = F.conv2d(x, laplacian_kernel_target, padding=1)
    x = x.clamp(min=0)
    x[x >= 0.1] = 1
    x[x < 0.1] = 0

    return x


class EdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, edge_factor=1.0):
        super(EdgeLoss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.edge_factor = edge_factor

    def get_boundary(self, x):
        laplacian_kernel_target = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).cuda(device=x.device)
        x = x.unsqueeze(1).float()
        x = F.conv2d(x, laplacian_kernel_target, padding=1)
        x = x.clamp(min=0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0

        return x

    def compute_edge_loss(self, logits, targets):
        bs = logits.size()[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        # print(boundary_targets.shape)
        logits = F.softmax(logits, dim=1).argmax(dim=1).squeeze(dim=1)
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        # print(boundary_pre)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        # print(boundary_pre)
        # dice_loss = 1 - ((2. * (boundary_pre * boundary_targets).sum(1) + 1.0) /
        #                  (boundary_pre.sum(1) + boundary_targets.sum(1) + 1.0))
        # dice_loss = dice_loss.mean()
        edge_loss = F.binary_cross_entropy_with_logits(boundary_pre, boundary_targets)

        return edge_loss

    def forward(self, logits, targets):
        loss = (self.main_loss(logits, targets) + self.compute_edge_loss(logits, targets) * self.edge_factor) / (self.edge_factor+1)
        return loss


class OHEM_CELoss(nn.Module):

    def __init__(self, thresh=0.7, ignore_index=255):
        super(OHEM_CELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_index].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class SIELoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
        # self.detail_loss = DetailAggregateLoss()
        self.CE = torch.nn.BCEWithLogitsLoss()



    def forward(self, logits, labels):
        if self.training and len(logits) == 8:
            # edge prediction
            edges = get_boundary(labels)
            # s1, s1_sig, edg1, s2, s2_sig, edg2, s3, s3_sig, edg3, s4, s4_sig, edg4 = logits
            s1,edg1, s2, edg2, s3, edg3, s4,  edg4 = logits
            # print(labels)
            loss1 = self.main_loss(s1, labels)  + self.CE(edg1, edges)
            loss2 = self.main_loss(s2, labels)  + self.CE(edg2, edges)
            loss3 = self.main_loss(s3, labels) + self.CE(edg3, edges)
            loss4 = self.main_loss(s4, labels) + self.CE(edg4, edges)
            loss = (loss1 + loss2 + loss3 + loss4)/4.0
            # boundery_bce_loss, boundery_dice_loss = self.detail_loss(logit_boundery, labels)
            # loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux, labels) + boundery_bce_loss + boundery_dice_loss
        else:
            loss = self.main_loss(logits, labels)

        return loss


class AFALoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
        # # self.detail_loss = DetailAggregateLoss()
        # self.CE = torch.nn.BCEWithLogitsLoss()



    def forward(self, logits, labels):
        if self.training and len(logits) == 4:
            s1, s2, s3, s4  = logits
            # print(labels)
            loss1 = self.main_loss(s1, labels)  
            loss2 = self.main_loss(s2, labels)  
            loss3 = self.main_loss(s3, labels) 
            loss4 = self.main_loss(s4, labels)
            loss = (loss1 + loss2 + loss3 + loss4)/4.0
        else:
            loss = self.main_loss(logits, labels)

        return loss
    
class AFALoss5(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
        # # self.detail_loss = DetailAggregateLoss()
        # self.CE = torch.nn.BCEWithLogitsLoss()



    def forward(self, logits, labels):
        if self.training and len(logits) == 5:
            x, s1, s2, s3, s4  = logits
            # print(labels)
            loss0 = self.main_loss(x, labels)  
            loss1 = self.main_loss(s1, labels)  
            loss2 = self.main_loss(s2, labels)  
            loss3 = self.main_loss(s3, labels) 
            loss4 = self.main_loss(s4, labels)
            loss = (loss0 + loss1 + loss2 + loss3 + loss4)/5.0
        else:
            loss = self.main_loss(logits, labels)

        return loss
    
class afaLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
        # self.detail_loss = DetailAggregateLoss()
        self.CE = torch.nn.BCEWithLogitsLoss()



    def forward(self, logits, labels):
        if self.training and len(logits) == 8:
            # edge prediction
            edges = get_boundary(labels)

            s1, s2, s3, s4, edg1, edg2, edg3, edg4 = logits
            # print(labels)
            loss1 = self.main_loss(s1, labels)  + self.CE(edg1, edges)
            loss2 = self.main_loss(s2, labels)  + self.CE(edg2, edges)
            loss3 = self.main_loss(s3, labels) + self.CE(edg3, edges)
            loss4 = self.main_loss(s4, labels) + self.CE(edg4, edges)
            loss = (loss1 + loss2 + loss3 + loss4)/4.0
        else:
            loss = self.main_loss(logits, labels)

        return loss

if __name__ == '__main__':
    targets = torch.randint(low=0, high=2, size=(2, 16, 16))
    logits = torch.randn((2, 2, 16, 16))
    # print(targets)
    model = EdgeLoss()
    loss = model.compute_edge_loss(logits, targets)

    print(loss)