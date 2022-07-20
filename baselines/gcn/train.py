"""
DGL Implementation of the GCN Paper.
[Semi-supervised classification with graph convolutional networks]
 (http://de.arxiv.org/pdf/1609.02907.pdf).
"""
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from gcn import GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN
import os
import random
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score,classification_report,precision_recall_curve
from sklearn.model_selection import train_test_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from dgl.data.utils import load_graphs, load_info
from imblearn.metrics import geometric_mean_score
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



def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)
    dgl.random.seed(seed)

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

# def gen_mask(g, train_rate, val_rate):
#     labels = g.ndata['label']
#     g.ndata['label'] = labels.long()
#     labels = np.array(labels)
#     n_nodes = len(labels)
#     index=list(range(n_nodes))
#     train_idx, val_test_idx, _, y_validate_test = train_test_split(index, labels, stratify=labels, train_size=train_rate,test_size=1-train_rate,
#                                                  random_state=2, shuffle=True)
#     val_idx, test_idx, _, _ = train_test_split(val_test_idx,y_validate_test, train_size=val_rate/(1-train_rate), test_size=1-val_rate/(1-train_rate),
#                                                      random_state=2, shuffle=True)
#     train_mask = torch.zeros(n_nodes, dtype=torch.bool)
#     val_mask = torch.zeros(n_nodes, dtype=torch.bool)
#     test_mask = torch.zeros(n_nodes, dtype=torch.bool)
#     train_mask[train_idx] = True
#     val_mask[val_idx] = True
#     test_mask[test_idx] = True
#     g.ndata['train_mask'] = train_mask
#     g.ndata['val_mask'] = val_mask
#     g.ndata['test_mask'] = test_mask
#     return g,train_idx

def gen_mask(g, train_rate, val_rate,IR,IR_set):
    labels = g.ndata['label']
    g.ndata['label'] = labels.long()
    labels = np.array(labels)
    n_nodes = len(labels)
    if IR_set==0:
        index=list(range(n_nodes))
    # 不平衡测试专用
    else:
        fraud_index=np.where(labels == 1)[0].tolist()
        benign_index = np.where(labels == 0)[0].tolist()
        if len(np.unique(labels))==3:
            Courier_index=np.where(labels == 2)[0].tolist()
        if IR<(len(fraud_index)/len(benign_index)):
            number_sample = int(IR * len(benign_index))
            sampled_fraud_index = random.sample(fraud_index, number_sample)
            sampled_benign_index=benign_index
            if len(np.unique(labels)) == 3:
                sampled_Courier_index=  random.sample(Courier_index, number_sample)
        else:
            number_sample = int( len(fraud_index)/IR)
            sampled_benign_index=random.sample(benign_index, number_sample)
            sampled_fraud_index=fraud_index
            if len(np.unique(labels)) == 3:
                sampled_Courier_index=Courier_index
        if len(np.unique(labels)) == 2:
            index = sampled_benign_index + sampled_fraud_index
        else:
            index = sampled_benign_index + sampled_fraud_index+sampled_Courier_index
        labels=labels[index]

    train_idx, val_test_idx, _, y_validate_test = train_test_split(index, labels, stratify=labels, train_size=train_rate,test_size=1-train_rate,
                                                 random_state=2, shuffle=True)
    val_idx, test_idx, _, _ = train_test_split(val_test_idx,y_validate_test, train_size=val_rate/(1-train_rate), test_size=1-val_rate/(1-train_rate),
                                                     random_state=2, shuffle=True)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    return g,train_idx

def main(args):
    # load and preprocess dataset
    if args.dataset == 'Sichuan':
        data, _ = load_graphs("../../data/Sichuan_tele.bin")  # glist will be [g1]
        num_classes = load_info("../../data/Sichuan_tele.pkl")['num_classes']
        g, _ = gen_mask(data[0], args.train_size, 0.2,args.IR,args.IR_set)

        g = g.to(device)
        features = g.ndata['feat'].float()
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = num_classes
        n_edges = g.number_of_edges()

    elif args.dataset == 'BUPT':
        data, _ = load_graphs("../../data/BUPT_tele.bin")  # glist will be [g1]
        num_classes = load_info("../../data/BUPT_tele.pkl")['num_classes']
        g,_ = gen_mask(data[0], args.train_size, 0.2,args.IR,args.IR_set)

        g = g.to(device)
        features = g.ndata['feat'].float()
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = num_classes
        n_edges = g.number_of_edges()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)


    if args.dataset == 'Sichuan' or 'BUPT':
        pass
    else:
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_labels
        n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)
    g = g.to(device)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout).to(device)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

    # test
    model.eval()
    output = model(features)
    logp = F.log_softmax(output, 1)
    loss_test = F.nll_loss(logp[test_mask], labels[test_mask])
    acc_test = accuracy(logp[test_mask], labels[test_mask])
    f1=f1_score(labels[test_mask].cpu().detach().numpy(), logp[test_mask].cpu().detach().numpy().argmax(axis=1), average='weighted')

    logp=torch.softmax(logp,dim=1)
    # calculate AUC
    if n_classes == 2:
        forest_auc = roc_auc_score(labels[test_mask].cpu().detach().numpy(), logp[test_mask].cpu().detach().numpy()[:,1],average='macro')
    else:
        forest_auc = roc_auc_score(labels[test_mask].cpu(), logp[test_mask].cpu().detach().numpy(), multi_class='ovo',average='macro')

    # print report
    target_names = ['{}'.format(i) for i in range(n_classes)]
    report = classification_report(labels[test_mask].cpu().detach().numpy(), logp[test_mask].cpu().detach().numpy().argmax(axis=1), target_names=target_names, digits=4)
    print("Test set results:\n",
          "idx_train{}".format(test_mask.shape),
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test),
          "F1= {:.4f}".format(f1.item()),
          "Auc= {:.4f}".format(forest_auc.item()),
          "\nReport=\n{}".format(report))

    test_gmean = geometric_mean_score(labels[test_mask].cpu().detach().numpy(), logp[test_mask].cpu().detach().numpy().argmax(axis=1))
    print("G mean={:.4f}".format(test_gmean))
    recall = recall_score(labels[test_mask].cpu().detach().numpy(), logp[test_mask].cpu().detach().numpy().argmax(axis=1), average='macro')
    logging.log(24, f"AUC:{forest_auc:.4f},F1:{f1:.4f},Recall:{recall:.4f},G-mean:{test_gmean:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="Sichuan",
                        help="Dataset name ('Sichuan','BUPT').")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    parser.add_argument('--train_size', type=float, default=0.2, help='train size.')
    parser.add_argument('--IR', type=float, default=0.1, help='imbalanced ratio.')
    parser.add_argument('--IR_set', type=int, default=0, help='whether to set imbalanced ratio,1 for set ,0 for not.')
    args = parser.parse_args()
    print(args)
    setup_seed(42)

    main(args)
