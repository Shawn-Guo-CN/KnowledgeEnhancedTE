"""
Contains various alignment model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tree_models import *


class RootAlign(nn.Module):
    def __init__(self, word_embedding, config):
        super(RootAlign, self).__init__()
        self.rnn = VanillaRecursiveNN(word_embedding, config['hidden_dim'], config['cuda_flag'])
        self.linear = nn.Linear(config['hidden_dim'] * 2, config['relation_num'])

    def forward(self, p_tree, h_tree):
        p_tree.postorder_traverse(self.rnn)
        h_tree.postorder_traverse(self.rnn)

        out = F.log_softmax(self.linear(F.sigmoid(
            torch.cat((p_tree.calculate_result, h_tree.calculate_result), 1))))
        return out
