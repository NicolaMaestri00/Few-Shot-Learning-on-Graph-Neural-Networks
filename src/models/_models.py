"""    Models    """

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import GCNConv, GraphConv, MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree



class nc_GCN(torch.nn.Module):
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


class DropBlock:
    def __init__(self, dropping_method: str, drop_rate: float = 0.5):
        super().__init__()
        self.dropping_method = dropping_method
        self.drop_rate = drop_rate

    def drop(self, x: torch.Tensor, edge_index):
        if self.drop_rate <= 0 or self.drop_rate >= 1:
            return x, edge_index  # No dropout

        if self.dropping_method == 'DropNode':
            mask = torch.bernoulli(torch.full((x.size(0), 1), 1 - self.drop_rate)).to(x.device)
            x = x * mask / (1 - self.drop_rate)  # Re-scale to preserve activation magnitude

        elif self.dropping_method == 'DropEdge':
            num_edges = edge_index.size(1)
            keep_edges = int(num_edges * (1 - self.drop_rate))
            perm = torch.randperm(num_edges)[:keep_edges]
            edge_index = edge_index[:, perm]
            edge_index = edge_index[:, edge_index[0, :].argsort()]

        elif self.dropping_method == 'DropAttributes':
            x = F.dropout(x, self.drop_rate)

        else:
            raise ValueError(f"Unknown dropping method: {self.dropping_method} |",
                             f"Available methods are ['DropNode', 'DropEdge', 'DropAttributes']")

        return x, edge_index


class nc_GNN_DropBlock(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropping_method='DropNode', drop_rate=0.5):
        super().__init__()
        self.drop_block = DropBlock(dropping_method, drop_rate)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.xavier_uniform_(param)
            elif name.endswith('bias'):
                torch.nn.init.zeros_(param)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        if self.training: x, edge_index = self.drop_block.drop(x, edge_index)
        x = self.conv2(x, edge_index).relu()
        out = self.conv3(x, edge_index)
        return x, out


class nc_GCN_DropMessage(MessagePassing):
    def __init__(self, in_channels, out_channels, drop_message=0):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.drop_rate = drop_message
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.xavier_uniform_(param)
            elif name.endswith('bias'):
                torch.nn.init.zeros_(param)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)
        return out + self.bias

    def message(self, x_j, norm):
        if 0 < self.drop_rate < 1:
            x_j = F.dropout(x_j, p=self.drop_rate, training=self.training)
        return norm.view(-1, 1) * x_j


class nc_GNN_DropMessage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, drop_rate=0.5):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = nc_GCN_DropMessage(hidden_channels, hidden_channels, drop_message=drop_rate)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.xavier_uniform_(param)
            elif name.endswith('bias'):
                torch.nn.init.zeros_(param)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        out = self.conv3(x, edge_index)

        return x, out


class gc_GCN(torch.nn.Module):
    ''' Graph Convolutional Network '''
    def __init__(self, in_channels, hidden_channels, num_classes, cs_reg=False):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr='add')
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr='add')
        self.conv3 = GraphConv(hidden_channels, hidden_channels, aggr='add')
        self.lin = Linear(hidden_channels, num_classes)
        self.cs_reg = cs_reg
        self.init_weights()

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch)
        if self.cs_reg:                                                           # Cosine Similarity Regularization
            x = F.normalize(x, p=2, dim=1)                                        # L2-normalize graph embeddings
            self.lin.weight.data = F.normalize(self.lin.weight.data, p=2, dim=1)  # L2-normalize class representatives
        out = self.lin(x)
        return x, out

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.xavier_uniform_(param)
            elif name.endswith('bias'):
                torch.nn.init.zeros_(param)


class gc_GNN_Dropout(torch.nn.Module):
    """ GNN with dropout before the classifier  """
    def __init__(self, in_channels, hidden_channels, num_classes, drop_rate=0.5):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr='add')
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr='add')
        self.conv3 = GraphConv(hidden_channels, hidden_channels, aggr='add')
        self.lin = Linear(hidden_channels, num_classes)
        self.drop_rate = drop_rate
        self.init_weights()

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch)
        out = self.lin(F.dropout(x, p=self.drop_rate, training=self.training))
        return x, out

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.xavier_uniform_(param)
            elif name.endswith('bias'):
                torch.nn.init.zeros_(param)


class gc_GNN_DropBlock(torch.nn.Module):
    """ GNN with DropBlock (DropNode, DropEdge, DropAttributes) """
    def __init__(self, in_channels, hidden_channels, num_classes, dropping_method='DropNode', drop_rate=0.5):
        super().__init__()
        self.drop_block = DropBlock(dropping_method, drop_rate)
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr='add')
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr='add')
        self.conv3 = GraphConv(hidden_channels, hidden_channels, aggr='add')
        self.lin = Linear(hidden_channels, num_classes)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.xavier_uniform_(param)
            elif name.endswith('bias'):
                torch.nn.init.zeros_(param)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        if self.training: x, edge_index = self.drop_block.drop(x, edge_index)
        x = self.conv2(x, edge_index).relu()
        if self.training: x, edge_index = self.drop_block.drop(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        out = self.lin(F.dropout(x, p=0.5, training=self.training))
        return x, out


class gc_GCN_DropMessage(MessagePassing):
    def __init__(self, in_channels, out_channels, drop_message=0):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.drop_rate = drop_message
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.xavier_uniform_(param)
            elif name.endswith('bias'):
                torch.nn.init.zeros_(param)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)
        return out + self.bias

    def message(self, x_j, norm):
        if 0 < self.drop_rate < 1:
            x_j = F.dropout(x_j, p=self.drop_rate, training=self.training)
        return norm.view(-1, 1) * x_j


class gc_GNN_DropMessage(torch.nn.Module):
    """ GNN with DropMessage """
    def __init__(self, in_channels, hidden_channels, num_classes, drop_rate=0.5):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr='add')
        self.conv2 = gc_GCN_DropMessage(hidden_channels, hidden_channels, drop_message=drop_rate)
        self.conv3 = gc_GCN_DropMessage(hidden_channels, hidden_channels, drop_message=drop_rate)
        self.lin = Linear(hidden_channels, num_classes)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.xavier_uniform_(param)
            elif name.endswith('bias'):
                torch.nn.init.zeros_(param)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.normalize(x, p=2, dim=1)
        self.lin.weight.data = F.normalize(self.lin.weight.data, p=2, dim=1)
        out = self.lin(F.dropout(x, p=0.5, training=self.training))
        return x, out


