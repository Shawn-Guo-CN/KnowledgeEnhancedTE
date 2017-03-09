import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RecursiveNN(nn.Module):
    def __init__(self, word_embedding):
        super(RecursiveNN, self).__init__()
        self.word_dim = word_embedding.embeddings.size(1)
        self.embedding = nn.Embedding(word_embedding.embeddings.size(0),
                                      self.word_dim)
        self.embedding.weight = nn.Parameter(word_embedding.embeddings)
        self.linear = nn.Linear(2 * self.word_dim, self.word_dim)

    def forward(self, node):
        if not node.val is None:
            node.calculate_result = self.embedding(Variable(torch.LongTensor([node.word_id])))
            return node.calculate_result
        else:
            assert len(node.children) == 2
            node.calculate_result = self.linear(torch.cat((node.children[0].calculate_result,
                                                          node.children[1].calculate_result), 1))
            return node.calculate_result