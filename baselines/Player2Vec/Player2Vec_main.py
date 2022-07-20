"""
Implementation of Player2Vec in DGFraudTF-2.
Key player identification in underground forums over attributed
 heterogeneous information network embedding framework.
Proceedings of the 28th ACM international conference on information
and knowledge management. 2019.
"""

import argparse
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import optimizers

import sys
sys.path.append('/code/imbalance')
from baselines.Player2Vec.Player2Vec import Player2Vec
from baselines.utils.data_loader import load_data_BUPT,load_data_Sichuan
from baselines.utils.utils import preprocess_adj, preprocess_feature, sample_mask
import torch
import os
from sklearn.metrics import f1_score, average_precision_score, \
    classification_report, precision_recall_curve, roc_auc_score,recall_score

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
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--dataset_str', type=str, default='Sichuan',
                    help="['Sichuan','BUPT']")
parser.add_argument('--train_size', type=float, default=0.2,
                    help='training set percentage')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=0.001,
                    help='weight decay')
parser.add_argument('--batch_size', type=int, default=30000)
parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--learning_rate', default=0.01,
                    help='the ratio of training set in whole dataset.')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units in GCN')
parser.add_argument('--lr', default=0.01, help='learning rate')

args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def Player2Vec_main(support: list,
                    features: tf.SparseTensor,
                    label: tf.Tensor,
                    masks: list,
                    args) -> None:
    """
    Main function to train, val and test the model

    :param support: a list of the sparse adjacency matrices
    :param features: node feature tuple for all nodes {coords, values, shape}
    :param label: the label tensor for all nodes
    :param masks: a list of mask tensors to obtain the train, val, test data
    :param args: additional parameters
    """
    model = Player2Vec(args.input_dim, args.nhid, args.output_dim, args)
    optimizer = optimizers.Adam(lr=args.lr)

    # train
    for epoch in tqdm(range(args.epochs)):
        with tf.GradientTape() as tape:
            train_loss, train_acc,_ = model([support, features, label, masks[0]])

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # validation
        val_loss, val_acc,_ = model([support, features, label, masks[1]])
        print(
            f"Epoch: {epoch:d}, train_loss: {train_loss:.4f}, "
            f"train_acc: {train_acc:.4f},"
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    # test
    test_loss, test_acc,logits = model([support, features, label, masks[2]])
    print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")

    # test
    logp = tf.nn.softmax(logits[masks[2]])
    num_classes = label.shape[1]

    # calculate AUC
    if np.isnan(logp.numpy()).any() == True:
        logp = torch.tensor(np.nan_to_num(logp.numpy()))
    else:
        pass
    if num_classes == 2:
        forest_auc = roc_auc_score(label.numpy()[masks[2]].argmax(axis=1), logp.numpy()[:, 1].reshape(-1, 1),
                                 average='macro')
    else:
        forest_auc = roc_auc_score(label.numpy()[masks[2]].argmax(axis=1), logp.numpy(),
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
    # load the data
    if args.dataset_str == 'BUPT':
        adj_list, features, [idx_train, _, idx_val, _, idx_test, _], y = \
            load_data_BUPT( path='../../data',train_size=0.2,val_size=0.2)
        args.nodes = features.shape[0]

    if args.dataset_str == 'Sichuan':
        adj_list, features, [idx_train, _, idx_val, _, idx_test, _], y = \
            load_data_Sichuan( path='../../data',train_size=0.2,val_size=0.2)
        args.nodes = features.shape[0]


    # convert to dense tensors
    train_mask = tf.convert_to_tensor(sample_mask(idx_train, y.shape[0]))
    val_mask = tf.convert_to_tensor(sample_mask(idx_val, y.shape[0]))
    test_mask = tf.convert_to_tensor(sample_mask(idx_test, y.shape[0]))
    label = tf.convert_to_tensor(y, dtype=tf.float32)

    # get sparse tuples
    features = preprocess_feature(features)
    supports = [preprocess_adj(adj) for adj in adj_list]

    # initialize the model parameters
    args.num_meta = len(supports)
    args.input_dim = features[2][1]
    args.output_dim = y.shape[1]
    args.train_size = len(idx_train)
    args.class_size = y.shape[1]
    args.num_features_nonzero = features[1].shape

    # get sparse tensors
    features = tf.cast(tf.SparseTensor(*features), dtype=tf.float32)
    supports = [tf.cast(tf.SparseTensor(*support), dtype=tf.float32) for
                support in supports]

    Player2Vec_main(supports, features, label,
                    [train_mask, val_mask, test_mask], args)
