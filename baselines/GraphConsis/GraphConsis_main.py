"""
Implementation of GraphConsis in DGFraudTF-2.
Alleviating the Inconsistency Problem of Applying Graph
Neural Network to Fraud Detection.
(http://de.arxiv.org/pdf/2005.00625.pdf)
"""

import argparse
import numpy as np
from collections import namedtuple
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import tensorflow as tf
import sys
sys.path.append('/code/imbalance')
from baselines.GraphConsis.GraphConsis import GraphConsis
from baselines.utils.data_loader import load_data_Sichuan,load_data_BUPT
from baselines.utils.utils import preprocess_feature
import torch
from sklearn.metrics import f1_score, average_precision_score, \
    classification_report, precision_recall_curve, roc_auc_score,recall_score
import scipy.sparse as spp

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import logging

import datetime
def beijing(sec,what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
logging.Formatter.converter = beijing

# logging configuration
log_name=(datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d')
logging.basicConfig(
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=24,
    filename=log_name+'.log',
    filemode='a'
    )


# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='BUPT', help=' Sichuan, BUPT')
parser.add_argument('--seed', type=int, default=717, help='random seed')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--train_size', type=float, default=0.8,
                    help='training set percentage')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units')
parser.add_argument('--sample_sizes', type=list, default=[5, 5],
                    help='number of samples for each layer')
parser.add_argument('--identity_dim', type=int, default=0,
                    help='dimension of context embedding')
parser.add_argument('--eps', type=float, default=0.001,
                    help='consistency score threshold ε')
args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def GraphConsis_main(neigh_dicts, features, labels, masks, num_classes, args):

    train_nodes = masks[0]
    val_nodes = masks[1]
    test_nodes = masks[2]

    # training
    def generate_training_minibatch(nodes_for_training, all_labels,
                                    batch_size, features):
        nodes_for_epoch = np.copy(nodes_for_training)
        ix = 0
        np.random.shuffle(nodes_for_epoch)
        while len(nodes_for_epoch) > ix + batch_size:
            mini_batch_nodes = nodes_for_epoch[ix:ix + batch_size]
            batch = build_batch(mini_batch_nodes, neigh_dicts,
                                args.sample_sizes, features)
            labels = all_labels[mini_batch_nodes]
            ix += batch_size
            yield (batch, labels)
        mini_batch_nodes = nodes_for_epoch[ix:-1]
        batch = build_batch(mini_batch_nodes, neigh_dicts,
                            args.sample_sizes, features)
        labels = all_labels[mini_batch_nodes]
        yield (batch, labels)

    model = GraphConsis(features.shape[-1], args.nhid,
                        len(args.sample_sizes), num_classes, len(neigh_dicts))
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch:d}: training...")
        minibatch_generator = generate_training_minibatch(train_nodes,
                                                          labels,
                                                          args.batch_size,
                                                          features)
        batchs = len(train_nodes) / args.batch_size
        for inputs, inputs_labels in tqdm(minibatch_generator, total=batchs):

            with tf.GradientTape() as tape:
                predicted = model(inputs, features)
                loss = loss_fn(tf.convert_to_tensor(inputs_labels.argmax(axis=1)), predicted)
                acc = accuracy_score(inputs_labels.argmax(axis=1),
                                     predicted.numpy().argmax(axis=1))
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            print(f" loss: {loss.numpy():.4f}, acc: {acc:.4f}")

        # validation
        print("Validating...")
        val_results = model(build_batch(val_nodes, neigh_dicts,
                                        args.sample_sizes, features), features)
        loss = loss_fn(tf.convert_to_tensor(labels[val_nodes].argmax(axis=1)), val_results)
        val_acc = accuracy_score(labels[val_nodes].argmax(axis=1),
                                 val_results.numpy().argmax(axis=1))
        print(f" Epoch: {epoch:d}, "
              f"loss: {loss.numpy():.4f}, "
              f"acc: {val_acc:.4f}")

    # testing
    print("Testing...")
    results = model(build_batch(test_nodes, neigh_dicts,
                                args.sample_sizes, features), features)
    test_acc = accuracy_score(labels[test_nodes].argmax(axis=1),
                              results.numpy().argmax(axis=1))
    print(f"Test acc: {test_acc:.4f}")

    logp = results
    num_classes=len(np.unique(labels.argmax(axis=1)))

    # calculate AUC
    if np.isnan(logp.numpy()).any() == True:
        logp = torch.tensor(np.nan_to_num(logp.numpy()))
    else:
        pass
    if num_classes == 2:
        forest_auc = roc_auc_score(labels[test_nodes].argmax(axis=1), logp.numpy()[:, 1].reshape(-1, 1),
                                 average='macro')
    else:
        forest_auc = roc_auc_score(labels[test_nodes].argmax(axis=1), logp.numpy(),
                                 average='macro', multi_class='ovo')

    # print report
    target_names=['{}'.format(i) for i in range(num_classes)]
    report = classification_report(labels[test_nodes].argmax(axis=1), torch.argmax(torch.from_numpy(logp.numpy()),dim=1), target_names=target_names, digits=4)
    print("Test set results:\n",
          "Auc= {:.4f}".format(forest_auc),
          "\nReport=\n{}".format(report))

    from imblearn.metrics import geometric_mean_score
    test_gmean = geometric_mean_score(labels[test_nodes].argmax(axis=1), torch.argmax(torch.from_numpy(logp.numpy()),dim=1))
    recall = recall_score(labels[test_nodes].argmax(axis=1), torch.argmax(torch.from_numpy(logp.numpy()),dim=1), average='macro')
    f1 = f1_score(labels[test_nodes].argmax(axis=1), torch.argmax(torch.from_numpy(logp.numpy()),dim=1),
                  average='macro')
    logging.log(24, f"AUC:{forest_auc:.4f},F1:{f1:.4f},Recall:{recall:.4f},G-mean:{test_gmean:.4f}")


def build_batch(nodes: list, neigh_dicts: dict, sample_sizes: list,
                features: np.array) -> [namedtuple]:
    """
    :param nodes: node ids
    :param neigh_dicts: BIDIRECTIONAL adjacency matrix in dict {node:[node]}
    :param sample_sizes: sample size for each layer
    :param features: 2d features of nodes
    :return a list of namedtuple minibatch
        "src_nodes": node ids to retrieve from raw feature and
        feed to the first layer
        "dstsrc2srcs": list of dstsrc2src matrices from last to first layer
        "dstsrc2dsts": list of dstsrc2dst matrices from last to first layer
        "dif_mats": list of dif_mat matrices from last to first layer
    """

    output = []
    for neigh_dict in neigh_dicts:
        dst_nodes = [nodes]
        dstsrc2dsts = []
        dstsrc2srcs = []
        dif_mats = []

        max_node_id = max(list(neigh_dict.keys()))

        for sample_size in reversed(sample_sizes):
            ds, d2s, d2d, dm = compute_diffusion_matrix(dst_nodes[-1],
                                                        neigh_dict,
                                                        sample_size,
                                                        max_node_id,
                                                        features
                                                        )
            dst_nodes.append(ds)
            dstsrc2srcs.append(d2s)
            dstsrc2dsts.append(d2d)
            dif_mats.append(dm)

        src_nodes = dst_nodes.pop()

        MiniBatchFields = ["src_nodes", "dstsrc2srcs",
                           "dstsrc2dsts", "dif_mats"]
        MiniBatch = namedtuple("MiniBatch", MiniBatchFields)
        output.append(MiniBatch(src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats))

    return output


def compute_diffusion_matrix(dst_nodes, neigh_dict, sample_size,
                             max_node_id, features):
    def calc_consistency_score(n, ns):
        # Equation 3 in the paper
        consis = tf.exp(-tf.pow(tf.norm(tf.tile([features[n]], [len(ns), 1]) -
                                        features[ns], axis=1), 2))
        consis = tf.where(consis > args.eps, consis, 0)
        return consis

    def sample(n, ns):
        if len(ns) == 0:
            return []
        consis = calc_consistency_score(n, ns)

        # Equation 4 in the paper,为适应BUPT修改replace=True
        prob = consis / tf.reduce_sum(consis)
        return np.random.choice(ns, min(len(ns), sample_size),
                                replace=True, p=prob)

    def vectorize(ns):
        v = np.zeros(max_node_id + 1, dtype=np.float32)
        v[ns] = 1
        return v

    # calculate in sparse matrix
    def dropcols_coo(M, idx_to_drop):
        idx_to_drop = np.unique(idx_to_drop)
        C = M.tocoo()
        keep = ~np.in1d(C.col, idx_to_drop)
        C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
        C.col -= idx_to_drop.searchsorted(C.col)  # decrement column indices
        C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
        return C.tocsr()


    i = np.array([])
    j = np.array([])
    k = 0
    for n in dst_nodes:
        nei = sample(n, neigh_dict[n])
        i = np.concatenate((i, np.array(k).repeat(len(nei)))).astype(int)
        j = np.concatenate((j, nei)).astype(int)
        k += 1
    data = np.ones_like(i)
    adj_mat_full = spp.coo_matrix((data, (i, j)), shape=(len(dst_nodes), max_node_id + 1))

    nonzero_cols_mask = np.array([False for _ in range(max_node_id + 1)])
    nonzero_cols_mask[np.unique(j)] = True

    del_column=[i for i in range(len(nonzero_cols_mask)) if nonzero_cols_mask[i]==False]

    adj_mat = dropcols_coo(adj_mat_full,np.array(del_column))


    adj_mat_sum = np.sum(adj_mat, axis=1)
    dif_mat = np.nan_to_num(adj_mat / adj_mat_sum)

    # compute dstsrc mappings
    src_nodes = np.arange(nonzero_cols_mask.size)[nonzero_cols_mask]
    dstsrc = np.union1d(dst_nodes, src_nodes)
    dstsrc2src = np.searchsorted(dstsrc, src_nodes)
    dstsrc2dst = np.searchsorted(dstsrc, dst_nodes)

    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat


if __name__ == "__main__":
    # load the data
    if args.dataset=='Sichuan':
        adj_list, features, [idx_train, _, idx_val, _, idx_test, _], y = \
            load_data_Sichuan( path='../../data',train_size=0.2,val_size=0.2)
        args.nodes = features.shape[0]

    if args.dataset=='BUPT':
        adj_list, features, [idx_train, _, idx_val, _, idx_test, _], y = \
            load_data_BUPT( path='../../data',train_size=0.6,val_size=0.2)
        args.nodes = features.shape[0]

    num_classes = len(np.unique(y.argmax(axis=1)))
    label = np.array([y]).reshape(-1,num_classes)

    features = preprocess_feature(features, to_tuple=False)
    features = np.array(features.todense())

    # Equation 2 in the paper
    features = np.concatenate((features,
                               np.random.rand(features.shape[0],
                                              args.identity_dim)), axis=1)

    neigh_dicts = []
    for net in adj_list:
        neigh_dict = {}
        for i in range(len(y)):
            neigh_dict[i] = []
        nodes1 = net.nonzero()[0]
        nodes2 = net.nonzero()[1]
        for node1, node2 in zip(nodes1, nodes2):
            neigh_dict[node1].append(node2)
        neigh_dicts.append({k: np.array(v, dtype=np.int64)
                            for k, v in neigh_dict.items()})

    GraphConsis_main(neigh_dicts, features, label,
                     [idx_train, idx_val, idx_test], num_classes, args)
