"""
Implementation of GEM in DGFraudTF-2.
Heterogeneous Graph Neural Networks for Malicious Account Detection
(http://de.arxiv.org/pdf/2002.12307.pdf)
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import argparse
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import optimizers

import sys
sys.path.append('/code/imbalance')
from baselines.GEM.GEM import GEM
from baselines.utils.data_loader import load_data_Sichuan,load_data_BUPT
from baselines.utils.utils import preprocess_feature, preprocess_adj
from sklearn.metrics import f1_score, average_precision_score, \
    classification_report, precision_recall_curve, roc_auc_score,recall_score
import torch
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
parser.add_argument('--dataset', type=str, default='Sichuan', help=' Sichuan, BUPT')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--train_size', type=float, default=0.2,
                    help='training set percentage')
parser.add_argument('--epochs', type=int, default=44,
                    help='Number of epochs to train.')
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--lr', default=0.002, help='learning rate')

# GEM
parser.add_argument('--hop', default=2,
                    help='number of hops of neighbors to be aggregated')
parser.add_argument('--output_dim', default=128, help='gem layer unit')

args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def GEM_main(supports: list, features: tf.SparseTensor,
             label: tf.Tensor, masks: list, args) -> None:
    """
    :param supports: a list of the sparse adjacency matrix
    :param features: the feature of the sparse tensor for all nodes
    :param label: the label tensor for all nodes
    :param masks: a list of mask tensors to obtain the train-val-test data
    :param args: additional parameters
    """
    model = GEM(args.input_dim, args.output_dim, args)
    optimizer = optimizers.Adam(lr=args.lr)

    # train
    for _ in tqdm(range(args.epochs)):
        with tf.GradientTape() as tape:
            train_loss, train_acc,_ = model(
                [supports, features, label, masks[0]])

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # validation
        val_loss, val_acc,_ = model([supports, features, label, masks[1]])
        print(
            f"train_loss: {train_loss:.4f},"
            f" train_acc: {train_acc:.4f},"
            f"val_loss: {val_loss:.4f},"
            f"val_acc: {val_acc:.4f}")

    # test
    test_loss, test_acc,logits = model([supports, features, label, masks[2]])
    print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")

    # test
    logp = logits
    num_classes = label.numpy().shape[1]

    # calculate AUC
    if np.isnan(logp.numpy()).any() == True:
        logp = torch.tensor(np.nan_to_num(logp.numpy()))
    else:
        pass
    if num_classes == 2:
        forest_auc = roc_auc_score(label.numpy()[masks[2]], logp.numpy()[:, 1].reshape(-1, 1),
                                 average='macro')
    else:
        forest_auc = roc_auc_score(label.numpy()[masks[2]], logp.numpy(),
                                 average='macro', multi_class='ovo')


    # print report
    target_names = ['{}'.format(i) for i in range(num_classes)]
    report = classification_report(label.numpy()[masks[2]].argmax(axis=1),
                                   torch.argmax(torch.from_numpy(logp.numpy()), dim=1), target_names=target_names,
                                   digits=4)
    print("Test set results:\n",
          "Auc= {:.4f}".format(forest_auc),
          "\nReport=\n{}".format(report))

    from imblearn.metrics import geometric_mean_score
    test_gmean = geometric_mean_score(label.numpy()[masks[2]].argmax(axis=1),
                                      torch.argmax(torch.from_numpy(logp.numpy()), dim=1))
    recall = recall_score(label.numpy()[masks[2]].argmax(axis=1),
                          torch.argmax(torch.from_numpy(logp.numpy()), dim=1), average='macro')
    f1 = f1_score(label.numpy()[masks[2]].argmax(axis=1), torch.argmax(torch.from_numpy(logp.numpy()), dim=1),
                  average='macro')
    logging.log(24, f"AUC:{forest_auc:.4f},F1:{f1:.4f},Recall:{recall:.4f},G-mean:{test_gmean:.4f}")


if __name__ == "__main__":

    if args.dataset == 'BUPT':
        adj_list, features, [idx_train, _, idx_val, _, idx_test, _], y = \
            load_data_BUPT( path='../../data',train_size=0.2,val_size=0.2)
        args.nodes = features.shape[0]

    if args.dataset == 'Sichuan':
        adj_list, features, [idx_train, _, idx_val, _, idx_test, _], y = \
            load_data_Sichuan( path='../../data',train_size=args.train_size,val_size=0.2)
        args.nodes = features.shape[0]

    # convert to dense tensors
    label = tf.convert_to_tensor(y, dtype=tf.float32)

    # initialize the model parameters
    args.input_dim = features.shape[1]
    args.nodes_num = features.shape[0]
    args.class_size = y.shape[1]
    args.train_size = len(idx_train)
    args.device_num = len(adj_list)

    features = preprocess_feature(features)
    supports = [preprocess_adj(adj) for adj in adj_list]

    # get sparse tensors
    features = tf.cast(tf.SparseTensor(*features), dtype=tf.float32)
    supports = [tf.cast(tf.SparseTensor(*support), dtype=tf.float32) for
                support in supports]

    masks = [idx_train, idx_val, idx_test]

    GEM_main(supports, features, label, masks, args)
