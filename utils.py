import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as spp
import math
import pandas as pd
from collections import Counter
from sklearn.utils import check_random_state, check_array



"""
	Utility functions to handle early stopping and mixed droupout and mixed liner.
"""

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None

    def step(self, acc, model, epoch):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)


def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices.astype(np.float32)),
            torch.FloatTensor(coo.data),
            coo.shape)


def matrix_to_torch(X):
    if spp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)

def misclassification_cost( y_true, y_pred,cost_table):
    """Appends misclassification costs to model predictions.
    Parameters
    ----------
    y_true : array-like of shape = [n_samples, 1]
             True class values.

    y_pred : array-like of shape = [n_samples, 1]
             Predicted class values.
    """
    df = pd.DataFrame({'row': y_pred, 'column': y_true})
    df = df.merge(cost_table, how='left', on=['row', 'column'])

    return df['cost'].values

# cost matrix
SET_COST_MATRIX_HOW = ('uniform', 'inverse', 'log1p-inverse')
def _set_cost_matrix(y,how: str = 'inverse'):
    """Set the cost matrix according to the 'how' parameter."""
    classes_, _y_encoded = np.unique(y, return_inverse=True)
    _encode_map = {c: np.where(classes_ == c)[0][0] for c in classes_}
    origin_distr_ = dict(Counter(_y_encoded))
    classes, origin_distr = _encode_map.values(), origin_distr_
    cost_matrix = []
    for c_pred in classes:
        cost_c = [
            origin_distr[c_pred] / origin_distr[c_actual]
            for c_actual in classes
        ]
        cost_c[c_pred] = 1
        cost_matrix.append(cost_c)
    if how == 'uniform':
        return np.ones_like(cost_matrix)
    elif how == 'inverse':
        return cost_matrix
    elif how == 'log1p-inverse':
        return np.log1p(cost_matrix)
    else:
        raise ValueError(
            f"When 'cost_matrix' is string, it should be"
            f" in {SET_COST_MATRIX_HOW}, got {how}."
        )

def cost_table_calc( cost_matrix):
    """Creates a table of values from the cost matrix.
    Write the matrix form cost matrix in the form of coordinates + cost value

    Parameters
    ----------
    cost_matrix : array-like of shape = [n_classes, n_classes]

    Returns
    -------
    df : dataframe of shape = [n_classes * n_classes, 3]

    """
    table = np.empty((0, 3))

    for (x, y), value in np.ndenumerate(cost_matrix):
        # table = np.vstack((table, np.array([x + 1, y + 1, value])))
        table = np.vstack((table, np.array([x , y , value])))

    return pd.DataFrame(table, columns=['row', 'column', 'cost'])

def _validate_cost_matrix(cost_matrix, n_classes):
    """validate the cost matrix."""
    cost_matrix = check_array(cost_matrix,
        ensure_2d=True, allow_nd=False,
        force_all_finite=True)
    if cost_matrix.shape != (n_classes, n_classes):
        raise ValueError(
            "When 'cost_matrix' is array-like, it should"
            " be of shape = [n_classes, n_classes],"
            " got shape = {0}".format(cost_matrix.shape)
        )
    return cost_matrix

