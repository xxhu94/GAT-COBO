"""
This code is due to Zhiqin Yang (@visitworld123) Yutong Deng (@yutongD),
Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection  in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2

Paper: 'Heterogeneous Graph Neural Networks for Malicious Account Detection'
Link: https://arxiv.org/pdf/2002.12307.pdf
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import tensorflow as tf
from tensorflow import keras
from baselines.layers.layers import GEMLayer

from baselines.utils.metrics import accuracy


class GEM(keras.Model):

    def __init__(self, input_dim, output_dim, args):
        super().__init__()

        self.nodes_num = args.nodes_num
        self.class_size = args.class_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device_num = args.device_num
        self.hop = args.hop
        self.zero_init = tf.keras.initializers.Zeros()
        self.h_0 = tf.Variable(
            initial_value=self.zero_init(
                shape=(self.nodes_num, self.output_dim),
                dtype=tf.float32)
        )

        # GEM layers initialization
        self.layers_ = []
        self.input_layer = GEMLayer(self.nodes_num, self.input_dim,
                                    self.output_dim, self.device_num)
        for _ in range(self.hop - 1):
            self.layers_.append(
                GEMLayer(self.nodes_num, self.input_dim, self.output_dim,
                         self.device_num))

        # logistic weights initialization
        self.x_init = tf.keras.initializers.GlorotUniform()
        self.u = tf.Variable(
            initial_value=self.x_init(shape=(self.output_dim, self.class_size),
                                      dtype=tf.float32), trainable=True)

    def call(self, inputs):

        """:param inputs include support, x, label, mask
        support means a list of the sparse adjacency Tensor
        x means feature
        label means label tensor
        mask means a list of mask tensors to obtain the train data
        """

        supports, x, label, idx_mask = inputs

        # forward propagation
        outputs = [self.input_layer((x, supports, self.h_0))]
        for layer in self.layers_:
            hidden = layer((x, supports, outputs[-1]))
            outputs.append(hidden)
        gem_out = outputs[-1]

        # get masked data
        masked_data = tf.gather(gem_out, idx_mask)
        masked_label = tf.gather(label, idx_mask)

        # Eq. (7) in paper
        logits = tf.nn.softmax(tf.matmul(masked_data, self.u))
        loss = -tf.reduce_sum(
            tf.math.log(tf.nn.sigmoid(masked_label * logits)))
        acc = accuracy(logits, masked_label)

        return loss, acc,logits
