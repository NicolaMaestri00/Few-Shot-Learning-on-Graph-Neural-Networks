"""    Loss functions    """

import torch
import torch.nn.functional as F


def entropy_regularization(logits):
    probs = F.softmax(logits, dim=1)                                # Compute probabilities
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)   # Compute entropy
    return entropy.mean()                                           # Average entropy over batch


class PWLoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super(PWLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        d = F.pairwise_distance(x1, x2, keepdim=True)
        loss = (1 - y) * d.pow(2) + y * F.relu(self.margin - d).pow(2)
        return loss.mean()


def nt_xent_loss(z1, z2, tau=0.5):
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    batch_size = z1.size(0)
    sim_matrix = torch.mm(z1, z2.t()) / tau
    labels = torch.arange(batch_size).to(z1.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
