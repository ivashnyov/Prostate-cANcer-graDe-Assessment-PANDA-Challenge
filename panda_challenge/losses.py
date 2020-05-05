import torch
from torch.nn.modules.loss import _Loss
import numpy as np


class QWKLoss(_Loss):
    def __init__(self, n_classes=6, eps=1e-8):
        super(QWKLoss, self).__init__()
        self.n_classes = n_classes
        self.eps = eps

    def forward(self, input, target):
        return kappa_loss(input, target, self.n_classes, self.eps)


def kappa_loss(output, target, n_classes=6, eps=1e-8):
    """
    QWK loss function as described in https://arxiv.org/pdf/1612.00775.pdf

    Arguments:
        p: a tensor with probability predictions, [batch_size, n_classes],
        y, a tensor with one-hot encoded class labels, [batch_size, n_classes]
    Returns:
        QWK loss
    """

    W = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = (i-j)**2 / (n_classes-1)**2
    output = torch.nn.Softmax(dim=1)(output)
    weights = torch.from_numpy(W.astype(np.float32)).cuda()
    observed = torch.matmul(target.t(), output)
    expected = torch.matmul(
        target.sum(dim=0).view(-1, 1),
        output.sum(dim=0).view(1, -1)) / observed.sum()

    return (weights*observed).sum() / ((weights*expected).sum() + eps)
