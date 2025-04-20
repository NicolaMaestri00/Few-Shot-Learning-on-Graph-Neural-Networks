import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, cs_reg=False):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.cs_reg = cs_reg
        self.init_weights()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        if self.cs_reg:                                                                       # Cosine Similarity Regularization
            x = F.normalize(x, p=2, dim=1)                                                    # L2-normalize node embeddings
            self.conv3.lin.weight.data = F.normalize(self.conv3.lin.weight.data, p=2, dim=1)  # L2-normalize class representatives
        out = self.conv3(x, edge_index)
        return x, out

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.xavier_uniform_(param)
            elif name.endswith('bias'):
                torch.nn.init.zeros_(param)
