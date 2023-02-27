import torch
import torch.nn as nn
import torch.cuda
import torch.optim
import torch.utils.data

from collections import OrderedDict


class AlphaDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, scale=True, weight=None, alpha=0.5,
                size_average=True, return_losses=False):
        super(AlphaDistillationLoss, self).__init__()
        self.temp = temperature
        self.scale = scale
        self.alpha = alpha
        self.size_average = size_average
        self.return_losses = return_losses

        # Don't scale losses because they will be combined later
        self._hard_loss = nn.CrossEntropyLoss(weight=weight,
                                            size_average=False)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, soft_targets, hard_targets, top=(1, 5)):
        # Figure soft target predictions
        _, preds = torch.max(soft_targets, dim=1, keepdim=False)

        # Calculate Cross Entropy with true targets
        hard_loss = self._hard_loss(inputs, hard_targets)

        # Calculate number of correct hard predictions
        hard = torch.nonzero(preds.ne(hard_targets).data)

        # Calculate Cross Entropy with soft targets
        hi_temp_inputs = self.log_softmax(inputs / self.temp)
        # Need high temperature probability distribution as target
        _soft_targets = self.softmax(soft_targets / self.temp)
        soft_cross_entropy = -(hi_temp_inputs * _soft_targets).sum(1)
        soft_loss = soft_cross_entropy.sum()

        unscaled_soft_loss = soft_loss.clone()
        # Scale to balance hard and soft loss functions
        if self.scale:
            soft_loss *= self.temp ** 2

        # Calculate soft targets Entropy
        soft_entropy = -1 * _soft_targets * torch.log(_soft_targets)
        soft_entropy[soft_entropy != soft_entropy] = 0
        soft_entropy = soft_entropy.sum(1)

        # Calculate Kullback-Leibler divergence
        soft_kl_divergence = soft_cross_entropy - soft_entropy

        # Calculate number of correct soft predictions
        soft = torch.nonzero(preds.eq(hard_targets).data)

        # Sum unscaled losses
        loss = sum([(1 - self.alpha) * soft_loss, self.alpha * hard_loss])
        if self.size_average:
            loss /= inputs.size(0)
        if self.return_losses:
            return loss, soft_loss, hard_loss
        else:
            return loss