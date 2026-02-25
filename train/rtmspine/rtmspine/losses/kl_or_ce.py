import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from mmpose.registry import MODELS

@MODELS.register_module()
class KLDoBCELoss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.
    Modified from `the official implementation 

    <https://github.com/leeyegy/SimCC>`_.

    This class is modified from original MMPose KLD loss 
    by adding BCELoss computing in case of zero output distributions.

    Args:
        beta (float): Temperature factor of Softmax. Default: 1.0.
        label_softmax (bool): Whether to use Softmax on labels.
            Default: False.
        label_beta (float): Temperature factor of Softmax on labels.
            Default: 1.0.
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        mask (list[int]): Index of masked keypoints.
        mask_weight (float): Weight of masked keypoints. Default: 1.0.
        bce_treshold (float): treshold for loss value to use BCE instead of KLD
    """

    def __init__(self,
                 beta=1.0,
                 label_softmax=False,
                 label_beta=10.0,
                 use_target_weight=True,
                 mask=None,
                 mask_weight=1.0,
                 bce_use_sigmoid = False,
                 bce_treshold = 10-4):
        super(KLDoBCELoss, self).__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.label_beta = label_beta
        self.use_target_weight = use_target_weight
        self.mask = mask
        self.mask_weight = mask_weight

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')

        self.use_sigmoid = bce_use_sigmoid
        _bce_loss = F.binary_cross_entropy if bce_use_sigmoid \
            else F.binary_cross_entropy_with_logits
        self.bce_loss = partial(_bce_loss, reduction='none')
        self.bce_treshold = bce_treshold


    def criterion_s_kld(self, dec_outs, labels):
        """Criterion function."""
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.label_beta, dim=1)

        loss = self.kl_loss(log_pt, labels)
        #print('premeanLOSS shape', self.kl_loss(log_pt, labels).shape) [94, H]
        #loss = torch.mean(self.kl_loss(log_pt, labels), dim=1)
        #print('LOSS shape', loss.shape) 94
        return loss
    
    def criterion_s_bce(self, dec_outs, labels):
        """Criterion function."""

        loss = self.bce_loss(dec_outs, labels)
        
        return loss
    
    def criterion(self, pred, target, treshold = 10e-4):

        '''
        pred [N, K, H or W]
        '''

        loss = []

        for sample, label in zip(pred, target):

            lesser_than = torch.prod(sample < treshold, axis = 1)
            lesser_indices = torch.nonzero(lesser_than.type(torch.bool)).flatten()
            greater_indices = torch.nonzero(~lesser_than.type(torch.bool)).flatten()

            loss_kld = self.criterion_s_kld(sample[greater_indices], label[greater_indices])
            loss_bce = self.criterion_s_bce(sample[lesser_indices], label[lesser_indices])

            loss_sample = torch.mean(torch.cat((loss_kld, loss_bce)), dim = 1)

            loss.append(loss_sample)

        loss = torch.cat(loss)

        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """
        N, K, _ = pred_simcc[0].shape
        loss = 0

        if self.use_target_weight:
            weight = target_weight.reshape(-1)
        else:
            weight = 1.

        for pred, target in zip(pred_simcc, gt_simcc):

            t_loss = self.criterion(pred, target, treshold = self.bce_treshold).mul(weight)

            if self.mask is not None:
                t_loss = t_loss.reshape(N, K)
                t_loss[:, self.mask] = t_loss[:, self.mask] * self.mask_weight

            loss = loss + t_loss.sum()

        return loss / K