'''    Loss functions    '''

import torch
import torch.nn.functional as F


def entropy_regularization(logits):
    probs = F.softmax(logits, dim=1)                                # Compute probabilities
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)   # Compute entropy
    return entropy.mean()                                           # Average entropy over batch
