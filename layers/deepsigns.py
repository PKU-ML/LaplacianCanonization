import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.mlp import MLP
from layers.gnns import GCN, GIN
import dgl


class GCNDeepSigns(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, k, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super(GCNDeepSigns, self).__init__()
        self.enc = GCN(in_channels, hidden_channels, out_channels, num_layers, use_bn=use_bn, dropout=dropout,
                       activation=activation)
        rho_dim = out_channels * k
        self.rho = MLP(rho_dim, hidden_channels, k, num_layers, use_bn=use_bn, dropout=dropout, activation=activation)
        self.k = k

    def forward(self, g, x):
        x = self.enc(g, x)  # we removed self.enc(g, -x)
        orig_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x = self.rho(x)
        x = x.reshape(orig_shape[0], self.k, 1)
        return x


class GINDeepSigns(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, k, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super(GINDeepSigns, self).__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers, use_bn=use_bn, dropout=dropout,
                       activation=activation)
        rho_dim = out_channels * k
        self.rho = MLP(rho_dim, hidden_channels, k, num_layers, use_bn=use_bn, dropout=dropout, activation=activation)
        self.k = k

    def forward(self, g, x):
        x = self.enc(g, x)  # we removed self.enc(g, -x)
        orig_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x = self.rho(x)
        x = x.reshape(orig_shape[0], self.k, 1)
        return x


class MaskedGINDeepSigns(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, k, device, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super(MaskedGINDeepSigns, self).__init__()
        self.device = device
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers, use_bn=use_bn, dropout=dropout,
                       activation=activation)
        self.rho = MLP(out_channels, hidden_channels, k, num_layers, use_bn=use_bn, dropout=dropout,
                       activation=activation)
        self.k = k

    def batched_n_nodes(self, n_nodes):
        t = torch.cat([size * torch.ones(size).to(self.device) for size in n_nodes])
        return t

    def forward(self, g, x):
        x = self.enc(g, x)  # we removed self.enc(g, -x)
        orig_shape = x.shape
        n_nodes = self.batched_n_nodes(g.batch_num_nodes().unsqueeze(1))
        mask = torch.cat([torch.arange(orig_shape[1]).unsqueeze(0) for i in range(orig_shape[0])])
        mask = (mask.to(self.device) < n_nodes.unsqueeze(1)).bool()
        x[~mask] = 0
        x = x.sum(dim=1)
        x = self.rho(x)
        x = x.reshape(orig_shape[0], self.k, 1)
        return x
