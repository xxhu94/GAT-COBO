import dgl
from dgl.data import DGLDataset
import torch
import numpy as np
import scipy.sparse as spp
from sklearn.preprocessing import StandardScaler
from dgl.data.utils import save_graphs, save_info


class TelcomFraudDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='telcom_fraud')

    def normalize(self,mx):
        # Row-normalize sparse matrix
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = spp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def process(self,path="./data/"):
        # load raw feature and labels
        idx_features_labels = np.genfromtxt("{}{}.csv".format(path, "all_feat_with_label"),
                                            dtype=np.dtype(str), delimiter=',', skip_header=1)
        features = spp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

        #normalize the feature with z-score
        features=StandardScaler().fit_transform(features.todense())
        labels = np.array(idx_features_labels[:, -1], dtype=np.int_)
        self.labels=torch.tensor(labels)
        node_features = torch.from_numpy(np.array(features))
        node_labels = torch.from_numpy(labels)

        # load adjacency matrix
        adj = spp.load_npz(path + 'node_adj_sparse.npz')
        adj = adj.toarray()
        adj = spp.coo_matrix(adj)

        # build symmetric adjacency matrix and normalize
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize(adj + spp.eye(adj.shape[0]))

        self.graph = dgl.from_scipy(adj)
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        # specify the default train,valid,test set for DGLgraph
        n_nodes = features.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        self.graph.num_labels = 2
        self._num_classes=2

        save_graphs('./data/Sichuan_tele.bin', self.graph, {'labels': self.labels})
        save_info('./data/Sichuan_tele.pkl', {'num_classes': self.num_classes})
        print('The Sichuan dataset is successfully generated! ')

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes


def BUPT_process():
    # load feature,labels and edges
    feature = np.genfromtxt("./data/TF.features", dtype=np.dtype(str), delimiter=' ')
    labels = np.genfromtxt("./data/TF.labels", dtype=np.dtype(str), delimiter=' ')
    edges = np.genfromtxt("./data/TF.edgelist", dtype=np.dtype(str), delimiter=' ')

    # normalize feature
    features = feature[:, 1:]
    features = features.astype(np.float32)
    normolize_features = StandardScaler().fit_transform(features)

    used_labels = labels[:len(feature), :]
    label_extract = used_labels[:, 1]
    label_extract = label_extract.reshape(len(label_extract), 1)

    node_num = used_labels[:, 0].astype(np.int32)
    node_num = list(node_num)
    node_new_num = [i for i in range(len(node_num))]
    edges = edges.astype(np.int32)
    src_edges = edges[:, 0]
    index = np.argwhere(src_edges < max(node_num) + 1)
    temp_edges = edges[0:max(index)[0] + 1, :]
    temp_dst_edges = temp_edges[:, 1]
    dst_edges_index = np.argwhere(temp_dst_edges < max(node_num) + 1)
    all_edges = temp_edges[dst_edges_index]
    all_edges = all_edges.reshape(len(all_edges), 2)

    src_e = all_edges[:, 0]
    dst_e = all_edges[:, 1]
    new_all_edges = np.empty_like(all_edges)
    np.copyto(new_all_edges, all_edges)
    # Update node numbers for all_edges
    for i, (src, dst) in enumerate(zip(src_e, dst_e)):
        src_start = 0 if src < 150 else src - 150
        dst_start = 0 if dst < 150 else dst - 150
        src_index = node_num.index(src, src_start)
        dst_index = node_num.index(dst, dst_start)
        new_all_edges[i] = [src_index, dst_index]


    triu_edges = [(edge[0], edge[1]) for i, edge in enumerate(new_all_edges) if edge[0] < edge[1]]
    symmetry_edges = np.empty_like(triu_edges)
    np.copyto(symmetry_edges, triu_edges)
    symmetry_edges[:, [0, -1]] = symmetry_edges[:, [-1, 0]]
    homo_graph_edges = np.concatenate((triu_edges, symmetry_edges))
    homo_graph_edges_unique = np.unique(homo_graph_edges, axis=0)
    selfloop_edges = [(node, node) for i, node in enumerate(node_new_num)]
    selfloop_edges = np.array(selfloop_edges)
    homo_graph_edges_unique_selfloop = np.concatenate((homo_graph_edges_unique, selfloop_edges))
    homo_graph = np.unique(homo_graph_edges_unique_selfloop, axis=0)

    src_id = homo_graph[:, 0]
    dst_id = homo_graph[:, 1]
    graph = dgl.graph((torch.tensor(src_id), torch.tensor(dst_id)))
    label_extract = label_extract.astype(np.int32).squeeze()
    graph.ndata['feat'] = torch.tensor(normolize_features)
    graph.ndata['label'] = torch.tensor(label_extract)
    save_graphs('./data/BUPT_tele.bin', graph, {'labels': torch.tensor(label_extract)})
    save_info('./data/BUPT_tele.pkl', {'num_classes': 3})
    print('The BUPT dataset is successfully generated! ')


if __name__=="__main__":
    # process Sichuan dataset
    dataset = TelcomFraudDataset()
    # process BUPT dataset
    BUPT_process()