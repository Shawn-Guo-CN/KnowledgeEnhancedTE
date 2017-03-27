"""
Tree models means the computation for every node in a tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VanillaRecursiveNN(nn.Module):
    def __init__(self, word_embedding, hidden_dim, cuda_flag=False):
        super(VanillaRecursiveNN, self).__init__()
        self.word_dim = word_embedding.embeddings.size(1)
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(word_embedding.embeddings.size(0),
                                      self.word_dim)
        self.embedding.weight = nn.Parameter(word_embedding.embeddings)

        self.word2hidden = nn.Linear(self.word_dim, self.hidden_dim, False)
        self.hidden2hidden = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        self.cuda_flag = cuda_flag

    def forward(self, node):
        if node.val is not None:
            if self.cuda_flag:
                node.calculate_result = self.word2hidden(self.embedding(
                    Variable(torch.LongTensor([node.word_id]).cuda())))
            else:
                node.calculate_result = self.word2hidden(self.embedding(
                    Variable(torch.LongTensor([node.word_id]))))
            return node.calculate_result
        else:
            assert len(node.children) == 2
            node.calculate_result = self.hidden2hidden(torch.cat((
                node.children[0].calculate_result, node.children[1].calculate_result), 1))
            return node.calculate_result


class BinaryTreeLSTM(nn.Module):
    def __init__(self, word_embedding, hidden_dim, cuda_flag=False):
        super(BinaryTreeLSTM, self).__init__()
        self.word_dim = word_embedding.embeddings.size(1)
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(word_embedding.embeddings.size(0),
                                      self.word_dim)
        self.embedding.weight = nn.Parameter(word_embedding.embeddings)

        self.word2hidden = nn.Linear(self.word_dim, self.hidden_dim, False)
        self.hidden2hidden = nn.Linear(self.hidden_dim, 5 * self.hidden_dim)

        self.cuda_flag = cuda_flag

    def forward(self, node):
        if not node.val is None:
            if self.cuda_flag:
                var = Variable(torch.LongTensor([node.word_id]).cuda())
            else:
                var = Variable(torch.LongTensor([node.word_id]))
            h = c = self.word2hidden(self.embedding(var))
            node.calculate_result = torch.cat((h, c), 1)
            return node.calculate_result
        else:
            assert len(node.children) == 2
            lh = node.children[0].calculate_result[0, :self.hidden_dim]
            lh.data.unsqueeze_(0)
            rh = node.children[1].calculate_result[0, :self.hidden_dim]
            rh.data.unsqueeze_(0)
            lc = node.children[0].calculate_result[0, self.hidden_dim:]
            lc.data.unsqueeze_(0)
            rc = node.children[1].calculate_result[0, self.hidden_dim:]
            rc.data.unsqueeze_(0)

            lo2g = self.hidden2hidden(lh)
            ro2g = self.hidden2hidden(rh)

            sum = lo2g + ro2g
            sigmoid_chunk = F.sigmoid(sum[0, :4 * self.hidden_dim])
            input_gate = sigmoid_chunk[:self.hidden_dim]
            input_gate.data.unsqueeze_(0)
            lf_gate = sigmoid_chunk[self.hidden_dim: 2 * self.hidden_dim]
            lf_gate.data.unsqueeze_(0)
            rf_gate = sigmoid_chunk[2 * self.hidden_dim: 3 * self.hidden_dim]
            rf_gate.data.unsqueeze(0)
            output_gate = sigmoid_chunk[3 * self.hidden_dim: 4 * self.hidden_dim]
            output_gate.data.unsqueeze_(0)
            hidden = F.tanh(sum[0, 4 * self.hidden_dim:])
            hidden.data.unsqueeze_(0)

            c = input_gate * hidden + lf_gate * lc + rf_gate * rc
            h = output_gate * F.tanh(c)

            node.calculate_result = torch.cat((h, c), 1)

            return node.calculate_result
