"""
Node model means similarity judger taking two node for input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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


class Node2TreeAttention(nn.Module):
    def __init__(self, node_hidden_size):
        super(Node2TreeAttention, self).__init__()
        self.hidden_size = node_hidden_size
        self.linear = nn.Linear(2 * self.hidden_size, 1, False)

    def forward(self, node):
        assert hasattr(self, 'tree_result')
        assert hasattr(self, 'num_nodes')

        v = node.calculate_result.data.expand(self.num_nodes, self.hidden_size)
        v = Variable(v)
        input = torch.cat((v, self.tree_result), 1)
        attn = F.softmax(torch.t(F.sigmoid(self.linear(input))))
        node.attn = attn.mm(self.tree_result)

        return node.attn

    def set_tree_result(self, tree):
        self.tree_result = tree.gather_calculate_result()
        self.num_nodes = tree.postorder_id + 1
