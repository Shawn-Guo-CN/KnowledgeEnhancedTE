"""
Contains various alignment model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tree_models import *
from node_model import *


class RootAlign(nn.Module):
    def __init__(self, word_embedding, config):
        super(RootAlign, self).__init__()
        self.name = 'RootAlign_vRNN'
        self.rnn = VanillaRecursiveNN(word_embedding, config['hidden_dim'], config['cuda_flag'])
        self.linear = nn.Linear(config['hidden_dim'] * 2, config['relation_num'])

    def forward(self, p_tree, h_tree):
        p_tree.postorder_traverse(self.rnn)
        h_tree.postorder_traverse(self.rnn)

        out = F.softmax(self.linear(torch.cat((
            p_tree.calculate_result, h_tree.calculate_result), 1)))
        return out

class RootAlign_BLSTM(nn.Module):
    def __init__(self, word_embedding, config):
        super(RootAlign_BLSTM, self).__init__()
        self.name = 'RootAlign_LSTM'
        self.rnn = BinaryTreeLSTM(word_embedding, config['hidden_dim'], config['cuda_flag'])
        self.linear = nn.Linear(config['hidden_dim'] * 2, config['relation_num'])

    def forward(self, p_tree, h_tree):
        p_tree.postorder_traverse(self.rnn)
        h_tree.postorder_traverse(self.rnn)

        out = F.softmax(self.linear(F.sigmoid(torch.cat((
            p_tree.calculate_result[1], h_tree.calculate_result[1]), 1))))
        return out

class Test(nn.Module):
    def __init__(self, word_embedding, config):
        super(Test, self).__init__()
        self.name = 'Test'
        self.rnn = VanillaRecursiveNN(word_embedding, config['hidden_dim'], config['cuda_flag'])
        self.linear = nn.Linear(config['hidden_dim'] * 2, config['relation_num'])
        self.node2tree = Node2Tree()

    def forward(self, p_tree, h_tree):
        p_tree.postorder_traverse(self.rnn)
        h_tree.postorder_traverse(self.rnn)

        self.node2tree.set_tree(p_tree)
        h_tree.postorder_traverse(self.node2tree)
