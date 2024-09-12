import torch.nn.functional as F
import torch


def cross_entropy_loss(output, target):
    """Takes output and target tensors and compute Cross Entropy Loss."""
    return F.cross_entropy(output, target)
