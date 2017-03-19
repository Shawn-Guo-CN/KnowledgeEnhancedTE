"""
Node model means similarity judger taking two node for input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nltk.corpus import wordnet as wn


class NeuralSimilarity(nn.Module):
    def __init__(self, node_hidden_size, out_size):
        super(NeuralSimilarity, self).__init__()
        self.linear = nn.Linear(2 * node_hidden_size, out_size)

    def forward(self, node1, node2):
        if isinstance(node1.calculate_result, Variable) and \
                isinstance(node2.calculate_result, Variable):
            result = self.linear(torch.cat((node1.calculate_result, node2.calculate_result), 1))
        else:
            assert len(node1.calculate_result) == len(node2.calculate_result)

            result = []
            for idx in xrange(len(node1.calculate_result)):
                result.append(self.linear(
                    torch.cat((node1.calculate_result[idx], node2.calculate_result[idx]), 1)))
        return result


class KnowledgeSimilarity(object):
    def __init__(self):
        pass

    def compute(self, node1, node2):
        def _word_compare_phrase(word_node, phrase_node):
            pass

        if node1.val is not None and node2.val is not None:
            pass
        elif node1.val is not None and node2.val is None:
            result = _word_compare_phrase(node1, node2)
        elif node1.val is None and node2.val is not None:
            result = _word_compare_phrase(node2, node1)
        else:
            assert node1.val is not None or node2.val is not None
            pass

        return result


class Node2Tree(nn.Module):
    def __init__(self):
        super(Node2Tree, self).__init__()

    def forward(self, node):
        assert hasattr(self, 'tree')
        print node.calculate_result.size()

    def set_tree(self, tree):
        self.tree = tree
