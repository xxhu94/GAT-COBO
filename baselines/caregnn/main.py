"""
DGL Implementation of the CARE-GNN Paper
This DGL example implements the CAmouflage-REsistant GNN (CARE-GNN)
model proposed in the paper [Enhancing Graph Neural Network-based
 Fraud Detectors against Camouflaged Fraudsters]
 (https://arxiv.org/abs/2008.08692).
"""
import dgl
import argparse
import torch as th
from model import CAREGNN
import torch.optim as optim
from sklearn.metrics import recall_score, roc_auc_score

from utils import EarlyStopping

import numpy as np
from sklearn.model_selection import train_test_split
import torch

from sklearn.metrics import f1_score, accuracy_score, average_precision_score, \
    classification_report, precision_recall_curve
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
#     return g


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

import random,os
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)
    dgl.random.seed(seed)

def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    from dgl.data.utils import load_graphs, load_info
    if args.dataset == 'BUPT':
        dataset, _ = load_graphs("../../data/BUPT_tele.bin")  # glist will be [g1]
        num_classes = load_info("../../data/BUPT_tele.pkl")['num_classes']
        graph,_ = gen_mask(dataset[0], args.train_size,0.2,args.IR,args.IR_set)

    elif args.dataset == 'Sichuan':
        dataset, _ = load_graphs("../../data/Sichuan_tele.bin")  # glist will be [g1]
        num_classes = load_info("../../data/Sichuan_tele.pkl")['num_classes']
        graph,_ = gen_mask(dataset[0], args.train_size,0.2,args.IR,args.IR_set)

    else:
        dataset = dgl.data.FraudDataset(args.dataset, train_size=0.4)
        graph = dataset[0]
        num_classes = dataset.num_classes


    # check cuda
    if args.gpu >= 0 and th.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    # retrieve labels of ground truth
    labels = graph.ndata['label'].long().squeeze().to(device)

    # Extract node features
    feat = graph.ndata['feat'].float().to(device)

    # retrieve masks for train/validation/test
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze(1).to(device)
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze(1).to(device)
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze(1).to(device)

    # Reinforcement learning module only for positive training nodes
    rl_idx = th.nonzero(train_mask.to(device) & labels.bool(), as_tuple=False).squeeze(1)

    graph = graph.to(device)

    # Step 2: Create model =================================================================== #
    model = CAREGNN(in_dim=feat.shape[-1],
                    num_classes=num_classes,
                    hid_dim=args.hid_dim,
                    num_layers=args.num_layers,
                    activation=th.tanh,
                    step_size=args.step_size,
                    edges=graph.canonical_etypes)

    model = model.to(device)

    # Step 3: Create training components ===================================================== #
    _, cnt = th.unique(labels, return_counts=True)
    loss_fn = th.nn.CrossEntropyLoss(weight=1 / cnt)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)

    # Step 4: training epochs =============================================================== #
    for epoch in range(args.max_epoch):
        # Training and validation using a full graph
        model.train()
        logits_gnn, logits_sim = model(graph, feat)

        # compute loss
        tr_loss = loss_fn(logits_gnn[train_idx], labels[train_idx]) + \
                  args.sim_weight * loss_fn(logits_sim[train_idx], labels[train_idx])

        tr_recall = recall_score(labels[train_idx].cpu(), logits_gnn.data[train_idx].argmax(dim=1).cpu(),average='weighted')

        # calculate train AUC
        if num_classes == 2:
            tr_auc = roc_auc_score(labels[train_idx].cpu(), th.softmax(logits_gnn.data[train_idx].cpu(),dim=1)[:,1],average='macro')
        else:
            tr_auc = roc_auc_score(labels[train_idx].cpu(), th.softmax(logits_gnn.data[train_idx].cpu(),dim=1),average='macro',multi_class='ovo')

        # validation
        val_loss = loss_fn(logits_gnn[val_idx], labels[val_idx]) + \
                   args.sim_weight * loss_fn(logits_sim[val_idx], labels[val_idx])
        val_recall = recall_score(labels[val_idx].cpu(), logits_gnn.data[val_idx].argmax(dim=1).cpu(),average='weighted')

        # calculate validation AUC
        if num_classes==2:
            val_auc = roc_auc_score(labels[val_idx].cpu(), th.softmax(logits_gnn.data[val_idx].cpu(),dim=1)[:, 1],average='macro')
        else:
            val_auc = roc_auc_score(labels[val_idx].cpu(), th.softmax(logits_gnn.data[val_idx].cpu(),dim=1),multi_class='ovo')

        # backward
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        # Print out performance
        print("Epoch {}, Train: Recall: {:.4f} AUC: {:.4f} Loss: {:.4f} | Val: Recall: {:.4f} AUC: {:.4f} Loss: {:.4f}"
              .format(epoch, tr_recall, tr_auc, tr_loss.item(), val_recall, val_auc, val_loss.item()))

        # Adjust p value with reinforcement learning module
        model.RLModule(graph, epoch, rl_idx)

        if args.early_stop:
            if stopper.step(val_auc, model):
                break

    # Test after all epoch
    model.eval()
    if args.early_stop:
        model.load_state_dict(th.load('es_checkpoint.pt'))

    # forward
    logits_gnn, logits_sim = model.forward(graph, feat)


    # test
    model.eval()
    logp = logits_gnn

    if np.isnan(logp[test_mask].cpu().detach().numpy()).any() == True:
        logp[test_mask] = torch.tensor(np.nan_to_num(logp[test_mask].cpu().detach().numpy())).to(device)
    else:
        pass

    if num_classes == 2:
        forest_auc = roc_auc_score(labels[test_mask].cpu().detach().numpy(), torch.softmax(logp[test_mask].cpu(), dim=1)[:, 1].detach().numpy(),
                                 average='macro')
    else:
        forest_auc = roc_auc_score(labels[test_mask].cpu().detach().numpy(), torch.softmax(logp[test_mask].cpu(), dim=1).detach().numpy(),
                                 average='macro', multi_class='ovo')

    target_names=['{}'.format(i) for i in range(num_classes)]
    report = classification_report(labels[test_idx].cpu().detach().numpy(), torch.argmax(logp[test_idx],dim=1).cpu().detach().numpy(), target_names=target_names, digits=4)
    print("Test set results:\n",
          "Auc= {:.4f}".format(forest_auc),
          "\nReport=\n{}".format(report))

    from imblearn.metrics import geometric_mean_score
    test_gmean = geometric_mean_score(labels[test_mask].cpu().detach().numpy(),
                                      logp[test_mask].cpu().detach().numpy().argmax(axis=1))
    print("G mean={:.4f}".format(test_gmean))

    f1=f1_score(labels[test_mask].cpu().detach().numpy(), logp[test_mask].cpu().detach().numpy().argmax(axis=1), average='macro')
    recall = recall_score(labels[test_mask].cpu().detach().numpy(),
                          logp[test_mask].cpu().detach().numpy().argmax(axis=1), average='macro')
    logging.log(24, f"AUC:{forest_auc:.4f},F1:{f1:.4f},Recall:{recall:.4f},G-mean:{test_gmean:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN-based Anti-Spam Model')
    parser.add_argument("--dataset", type=str, default="BUPT", help="DGL dataset for this model (Sichuan, or BUPT)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Default: -1, using CPU.")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers")
    parser.add_argument("--max_epoch", type=int, default=30, help="The max number of epochs. Default: 30")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate. Default: 0.01")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay. Default: 0.001")
    parser.add_argument("--step_size", type=float, default=0.02, help="RL action step size (lambda 2). Default: 0.02")
    parser.add_argument("--sim_weight", type=float, default=2, help="Similarity loss weight (lambda 1). Default: 2")
    parser.add_argument('--early-stop', action='store_true', default=False, help="indicates whether to use early stop")
    parser.add_argument('--train_size', type=float, default=0.2, help='train size.')
    parser.add_argument('--IR', type=float, default=0.1, help='imbalanced ratio.')
    parser.add_argument('--IR_set', type=int, default=0, help='whether to set imbalanced ratio,1 for set ,0 for not.')
    args = parser.parse_args()
    print(args)
    setup_seed(42)
    main(args)
