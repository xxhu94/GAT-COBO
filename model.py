
import torch
import time
from dgl.nn import GATConv
from utils import MixedDropout, MixedLinear
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
import dgl,os

class GAT_COBO(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 dropout,
                 dropout_adj,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT_COBO, self).__init__()
        # MixedLinear
        fcs = [MixedLinear(in_dim, num_hidden, bias=False)]
        fcs.append(nn.Linear(num_hidden, num_classes, bias=False))

        self.fcs = nn.ModuleList(fcs)
        self.reg_params = list(self.fcs[0].parameters())
        if dropout is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(dropout)
        if dropout_adj is 0:
            self.dropout_adj = lambda x: x
        else:
            self.dropout_adj = MixedDropout(dropout_adj)
        self.act_fn = nn.ReLU()

        # GAT-based weak-classifier
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation,bias=False))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation,bias=False))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def transform_features(self, x):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(x)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.act_fn(self.fcs[-1](self.dropout_adj(layer_inner)))
        return res

    def forward(self, inputs):
        logits_inter_GAT = self.transform_features(inputs)
        h = inputs
        for l in range(self.num_layers):
            h= self.gat_layers[l](self.g, h).flatten(1)
        logits_inner_GAT, attention = self.gat_layers[-1](self.g, h,True)
        return logits_inter_GAT,logits_inner_GAT, attention