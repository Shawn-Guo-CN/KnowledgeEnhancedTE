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
        if not node.val is None:
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
                node.calculate_result = self.word2hidden(
                    self.embedding(Variable(torch.LongTensor([node.word_id]).cuda())))
            else:
                node.calculate_result = self.word2hidden(
                    self.embedding(Variable(torch.LongTensor([node.word_id]))))
            return node.calculate_result
        else:
            assert len(node.children) == 2
            lo2g = self.hidden2hidden(node.children[0].calculate_result)
            ro2g = self.hidden2hidden(node.children[1].calculate_result)
            sum = lo2g + ro2g
            sigmoid_chunk = F.sigmoid(sum[:4 * self.hidden_dim])
            input_gate = sigmoid_chunk[:self.hidden_dim]
            lf_gate = sigmoid_chunk[self.hidden_dim: 2 * self.hidden_dim]
            rf_gate = sigmoid_chunk[2 * self.hidden_dim: 3 * self.hidden_dim]
            output_gate = sigmoid_chunk[3 * self.hidden_dim: 4 * self.hidden_dim]
            hidden = F.tanh(sum[4 * self.hidden_dim:])

            c = input_gate * hidden + lf_gate * node.children[0].calculate_result + \
                rf_gate * node.children[0].calculate_result

            node.calculate_result = self.hidden2hidden(torch.cat((node.children[0].calculate_result,
                                                          node.children[1].calculate_result), 1))
            return node.calculate_result
