from typing import Tuple
import numpy as np
import scipy.sparse as spp
import scipy.io as sio
from sklearn.model_selection import train_test_split
import tensorflow as tf
from baselines.utils.utils import pad_adjlist
from dgl.data.utils import load_graphs
from dgl.data.utils import makedirs, save_info, load_info



def load_data_BUPT(path: str = '../../data',
                   train_size: float = 0.2,val_size: float = 0.2) -> \
        Tuple[list, np.array, list, np.array]:

    dataset, _ = load_graphs(path+'/BUPT_tele.bin')  # glist will be [g1]
    n_classes = load_info(path+'/BUPT_tele.pkl')['num_classes']
    graph = dataset[0]
    features=graph.ndata['feat'].numpy()

    rownetworks=graph.adj_sparse('coo')
    i=rownetworks[0].numpy()
    j=rownetworks[1].numpy()
    data=np.ones_like(i)

    rownetworks=spp.coo_matrix((data,(i,j)),shape=(features.shape[0],features.shape[0]))

    y = graph.ndata['label'].numpy()
    y = tf.one_hot(y,n_classes)
    y = y.numpy()
    index = np.arange(len(y))
    X_train_val, X_test, y_train_val, y_test = \
        train_test_split(index, y, stratify=y, test_size=1 - train_size-val_size,
                         random_state=48, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      stratify=y_train_val,
                                                      train_size=train_size/(train_size+val_size),
                                                      random_state=48,
                                                      shuffle=True)

    split_ids = [X_train, y_train, X_val, y_val, X_test, y_test]

    return [rownetworks], features, split_ids, np.array(y)

def load_data_Sichuan(path: str = '../../dataset',
                   train_size: float = 0.2,val_size: float = 0.2
                      ) -> \
        Tuple[list, np.array, list, np.array]:

    dataset, _ = load_graphs(path+'/Sichuan_tele.bin')  # glist will be [g1]
    n_classes = load_info(path+'/Sichuan_tele.pkl')['num_classes']
    graph = dataset[0]
    features=graph.ndata['feat'].numpy()
    rownetworks=graph.adj_sparse('coo')

    i=rownetworks[0].numpy()
    j=rownetworks[1].numpy()
    data=np.ones_like(i)
    rownetworks=spp.coo_matrix((data,(i,j)),shape=(features.shape[0],features.shape[0]))


    y = graph.ndata['label'].numpy()
    y = tf.one_hot(y,n_classes)
    y = y.numpy()
    index = np.arange(len(y))
    X_train_val, X_test, y_train_val, y_test = \
        train_test_split(index, y, stratify=y, test_size=1 - train_size-val_size,
                         random_state=48, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      stratify=y_train_val,
                                                      train_size=train_size/(train_size+val_size),
                                                      random_state=48,
                                                      shuffle=True)

    split_ids = [X_train, y_train, X_val, y_val, X_test, y_test]

    return [rownetworks], features, split_ids, np.array(y)

if __name__=="__main__":
    load_data_BUPT()
    load_data_Sichuan()
