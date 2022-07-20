"""
Implementation of GraphSage in DGFraudTF-2.
Inductive Representation Learning on Large Graphs.
(http://de.arxiv.org/pdf/1706.02216.pdf)
"""

import argparse
import numpy as np
import collections
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tensorflow as tf

from baselines.GraphSage.GraphSage import GraphSage
from baselines.utils.data_loader import load_data_BUPT,load_data_Sichuan
from baselines.utils.utils import preprocess_feature
from sklearn.metrics import f1_score, average_precision_score, \
    classification_report, precision_recall_curve, roc_auc_score

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='BUPT', help=' Sichuan, BUPT')
parser.add_argument('--seed', type=int, default=717, help='random seed')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--train_size', type=float, default=0.8,
                    help='training set percentage')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units')
parser.add_argument('--sample_sizes', type=list, default=[5, 5],
                    help='number of samples for each layer')
args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def GraphSage_main(neigh_dict, features, labels, masks, num_classes, args):
    train_nodes = masks[0]
    val_nodes = masks[1]
    test_nodes = masks[2]

    # training
    def generate_training_minibatch(nodes_for_training,
                                    all_labels, batch_size):
        nodes_for_epoch = np.copy(nodes_for_training)
        ix = 0
        np.random.shuffle(nodes_for_epoch)
        while len(nodes_for_epoch) > ix + batch_size:
            mini_batch_nodes = nodes_for_epoch[ix:ix + batch_size]
            batch = build_batch(mini_batch_nodes,
                                neigh_dict, args.sample_sizes)
            labels = all_labels[mini_batch_nodes]
            ix += batch_size
            yield (batch, labels)
        mini_batch_nodes = nodes_for_epoch[ix:-1]
        batch = build_batch(mini_batch_nodes, neigh_dict, args.sample_sizes)
        labels = all_labels[mini_batch_nodes]
        yield (batch, labels)

    model = GraphSage(features.shape[-1], args.nhid,
                      len(args.sample_sizes), num_classes)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    # loss_fn = tf.keras.losses.CategoricalCrossentropy()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch:d}: training...")
        minibatch_generator = generate_training_minibatch(
            train_nodes, labels, args.batch_size)
        for inputs, inputs_labels in tqdm(
                minibatch_generator, total=len(train_nodes) / args.batch_size):
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
        val_results = model(build_batch(
            val_nodes, neigh_dict, args.sample_sizes), features)
        loss = loss_fn(tf.convert_to_tensor(labels[val_nodes].argmax(axis=1)), val_results)
        val_acc = accuracy_score(labels[val_nodes].argmax(axis=1),
                                 val_results.numpy().argmax(axis=1))
        print(f"Epoch: {epoch:d}, "
              f"loss: {loss.numpy():.4f}, "
              f"acc: {val_acc:.4f}")

    # testing
    print("Testing...")
    results = model(build_batch(
        test_nodes, neigh_dict, args.sample_sizes), features)
    test_acc = accuracy_score(labels[test_nodes].argmax(axis=1),
                              results.numpy().argmax(axis=1))
    print(f"Test acc: {test_acc:.4f}")

    logp = results
    num_classes = labels.shape[1]

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
    target_names = ['{}'.format(i) for i in range(num_classes)]
    report = classification_report(labels[test_nodes].argmax(axis=1),
                                   torch.argmax(torch.from_numpy(logp.numpy()), dim=1), target_names=target_names,
                                   digits=4)
    print("Test set results:\n",
          "Auc= {:.4f}".format(forest_auc),
          "\nReport=\n{}".format(report))
    from imblearn.metrics import geometric_mean_score
    test_gmean = geometric_mean_score(labels[test_nodes].argmax(axis=1), torch.argmax(torch.from_numpy(logp.numpy()), dim=1))
    print("G mean={:.4f}".format(test_gmean))



def build_batch(nodes, neigh_dict, sample_sizes):
    """
    :param [int] nodes: node ids
    :param {node:[node]} neigh_dict: BIDIRECTIONAL adjacency matrix in dict
    :param [sample_size]: sample sizes for each layer,
    lens is the number of layers
    :param tensor features: 2d features of nodes
    :return namedtuple minibatch
        "src_nodes": node ids to retrieve from raw feature
        and feed to the first layer
        "dstsrc2srcs": list of dstsrc2src matrices from last to first layer
        "dstsrc2dsts": list of dstsrc2dst matrices from last to first layer
        "dif_mats": list of dif_mat matrices from last to first layer
    """

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
                                                    )
        dst_nodes.append(ds)
        dstsrc2srcs.append(d2s)
        dstsrc2dsts.append(d2d)
        dif_mats.append(dm)

    src_nodes = dst_nodes.pop()

    MiniBatchFields = ["src_nodes", "dstsrc2srcs", "dstsrc2dsts", "dif_mats"]
    MiniBatch = collections.namedtuple("MiniBatch", MiniBatchFields)

    return MiniBatch(src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats)


def compute_diffusion_matrix(dst_nodes, neigh_dict, sample_size, max_node_id):
    def sample(ns):
        return np.random.choice(ns, min(len(ns), sample_size), replace=False)

    def vectorize(ns):
        v = np.zeros(max_node_id + 1, dtype=np.float32)
        v[ns] = 1
        return v

    # sample neighbors
    adj_mat_full = np.stack([vectorize(
        sample(neigh_dict[n])) for n in dst_nodes])
    nonzero_cols_mask = np.any(adj_mat_full.astype(np.bool), axis=0)

    # compute diffusion matrix
    adj_mat = adj_mat_full[:, nonzero_cols_mask]
    adj_mat_sum = np.sum(adj_mat, axis=1, keepdims=True)
    dif_mat = np.nan_to_num(adj_mat / adj_mat_sum)

    # compute dstsrc mappings
    src_nodes = np.arange(nonzero_cols_mask.size)[nonzero_cols_mask]
    # np.union1d automatic sorts the return,
    # which is required for np.searchsorted
    dstsrc = np.union1d(dst_nodes, src_nodes)
    dstsrc2src = np.searchsorted(dstsrc, src_nodes)
    dstsrc2dst = np.searchsorted(dstsrc, dst_nodes)

    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat


if __name__ == "__main__":
    # load the data

    if args.dataset=='BUPT':
        adj_list, features, [idx_train, _, idx_val, _, idx_test, _], y = \
            load_data_BUPT( path='../../data',train_size=0.6,val_size=0.2)
        args.nodes = features.shape[0]

    if args.dataset=='Sichuan':
        adj_list, features, [idx_train, _, idx_val, _, idx_test, _], y = \
            load_data_Sichuan( path='../../data',train_size=0.4)
        args.nodes = features.shape[0]

    num_classes = len(np.unique(y.argmax(axis=1)))
    label = np.array([y]).reshape(-1,num_classes)


    features = preprocess_feature(features, to_tuple=False)
    features = np.array(features.todense())

    neigh_dict = collections.defaultdict(list)
    for i in range(len(y)):
        neigh_dict[i] = []

    # merge all relations into single graph
    for net in adj_list:
        nodes1 = net.nonzero()[0]
        nodes2 = net.nonzero()[1]
        for node1, node2 in zip(nodes1, nodes2):
            neigh_dict[node1].append(node2)

    neigh_dict = {k: np.array(v, dtype=np.int64)
                  for k, v in neigh_dict.items()}

    GraphSage_main(neigh_dict, features, label,
                   [idx_train, idx_val, idx_test], num_classes, args)
