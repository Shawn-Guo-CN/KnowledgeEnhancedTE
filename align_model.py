"""
Contains various alignment model
"""

from tree_models import *
from node_model import *


class RootAlign(nn.Module):
    def __init__(self, word_embedding, config):
        super(RootAlign, self).__init__()
        self.name = 'RootAlign_vRNN'
        self.rnn = VanillaRecursiveNN(word_embedding, config['hidden_dim'], config['cuda_flag'])
        self.linear = nn.Linear(config['hidden_dim'] * 2, config['relation_num'])
        self.dropout = nn.Dropout(p=config['drop_p'])

    def forward(self, p_tree, h_tree):
        p_tree.postorder_traverse(self.rnn)
        h_tree.postorder_traverse(self.rnn)

        out = F.log_softmax(self.dropout(F.sigmoid(self.linear(torch.cat((
            p_tree.calculate_result, h_tree.calculate_result), 1)))))
        return out

class RootAlign_BLSTM(nn.Module):
    def __init__(self, word_embedding, config):
        super(RootAlign_BLSTM, self).__init__()
        self.name = 'RootAlign_LSTM'
        self.rnn = BinaryTreeLSTM(word_embedding, config['hidden_dim'], config['cuda_flag'])
        self.linear = nn.Linear(config['hidden_dim'] * 4, config['relation_num'])
        self.dropout = nn.Dropout(p=config['drop_p'])

    def forward(self, p_tree, h_tree):
        p_tree.postorder_traverse(self.rnn)
        h_tree.postorder_traverse(self.rnn)

        out = F.log_softmax(self.dropout(F.sigmoid(self.linear(torch.cat((
            p_tree.calculate_result, h_tree.calculate_result), 1)))))
        return out


class AttentionFromH2P_vRNN(nn.Module):
    def __init__(self, word_embedding, config):
        super(AttentionFromH2P_vRNN, self).__init__()
        self.name = 'AttentionFromH2P_vRNN'
        self.rnn = VanillaRecursiveNN(word_embedding, config['hidden_dim'], config['cuda_flag'])
        self.linear = nn.Linear(config['hidden_dim'] * 3, config['relation_num'], False)
        self.node2tree = Node2TreeAttention(config['hidden_dim'])
        self.dropout = nn.Dropout(p=config['drop_p'])

    def forward(self, p_tree, h_tree):
        p_tree.postorder_traverse(self.rnn)
        h_tree.postorder_traverse(self.rnn)

        self.node2tree.set_tree_result(p_tree)
        h_tree.postorder_traverse(self.node2tree)

        result = h_tree.get_attention_representation()
        result = torch.cat((h_tree.calculate_result, p_tree.calculate_result, result), 1)
        result = F.sigmoid(F.elu(self.linear(result)))

        out = F.softmax(self.dropout(result))

        return out

class DoubleAttention_vRNN(nn.Module):
    def __init__(self, word_embedding, config):
        super(DoubleAttention_vRNN, self).__init__()
        self.name = 'DoubleAttention_vRNN'
        self.rnn = VanillaRecursiveNN(word_embedding, config['hidden_dim'], config['cuda_flag'])
        self.linear = nn.Linear(config['hidden_dim'] * 4, config['relation_num'])
        self.node2tree = Node2TreeAttention(config['hidden_dim'])
        self.dropout = nn.Dropout(p=config['drop_p'])

    def forward(self, p_tree, h_tree):
        p_tree.postorder_traverse(self.rnn)
        h_tree.postorder_traverse(self.rnn)

        self.node2tree.set_tree_result(p_tree)
        h_tree.postorder_traverse(self.node2tree)

        self.node2tree.set_tree_result(h_tree)
        p_tree.postorder_traverse(self.node2tree)

        result1 = h_tree.get_attention_representation()
        result2 = p_tree.get_attention_representation()
        result = torch.cat((h_tree.calculate_result, p_tree.calculate_result, result1, result2), 1)
        result = F.sigmoid(F.elu(self.linear(result)))

        out = F.softmax(self.dropout(result))

        return out

class AttentionFromH2P_LSTM(nn.Module):
    def __init__(self, word_embedding, config):
        super(AttentionFromH2P_LSTM, self).__init__()
        self.name = 'AttentionFromH2P_LSTM'
        self.rnn = BinaryTreeLSTM(word_embedding, config['hidden_dim'], config['cuda_flag'])
        self.linear = nn.Linear(config['hidden_dim'] * 6, config['relation_num'], False)
        self.node2tree = Node2TreeAttention(2 * config['hidden_dim'])
        self.dropout = nn.Dropout(p=config['drop_p'])

    def forward(self, p_tree, h_tree):
        p_tree.postorder_traverse(self.rnn)
        h_tree.postorder_traverse(self.rnn)

        self.node2tree.set_tree_result(p_tree)
        h_tree.postorder_traverse(self.node2tree)

        result = h_tree.get_attention_representation()
        result = torch.cat((h_tree.calculate_result, p_tree.calculate_result, result), 1)
        result = F.sigmoid(F.elu(self.linear(result)))

        out = F.softmax(self.dropout(result))

        return out

class DoubleAttention_LSTM(nn.Module):
    def __init__(self, word_embedding, config):
        super(DoubleAttention_LSTM, self).__init__()
        self.name = 'DoubleAttention_LSTM'
        self.rnn = BinaryTreeLSTM(word_embedding, config['hidden_dim'], config['cuda_flag'])
        self.linear = nn.Linear(config['hidden_dim'] * 8, config['relation_num'])
        self.node2tree = Node2TreeAttention(2 * config['hidden_dim'])
        self.dropout = nn.Dropout(p=config['drop_p'])

    def forward(self, p_tree, h_tree):
        p_tree.postorder_traverse(self.rnn)
        h_tree.postorder_traverse(self.rnn)

        self.node2tree.set_tree_result(p_tree)
        h_tree.postorder_traverse(self.node2tree)

        self.node2tree.set_tree_result(h_tree)
        p_tree.postorder_traverse(self.node2tree)

        result1 = h_tree.get_attention_representation()
        result2 = p_tree.get_attention_representation()
        result = torch.cat((h_tree.calculate_result, p_tree.calculate_result, result1, result2), 1)
        result = F.sigmoid(F.elu(self.linear(result)))

        out = F.softmax(self.dropout(result))

        return out