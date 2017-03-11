import torch
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import *
from wordembedding import WordEmbedding
from snli import SNLI
from simpleprofiler import SimpleProfiler
from align_model import RootAlign


def train():
    word_embedding = WordEmbedding('./sampledata/wordembedding')
    snli = SNLI('./sampledata/')

    printerr("Before trim word embedding, " + str(word_embedding.embeddings.size(0)) + " words")
    word_embedding.trim_by_counts(snli.word_counts)
    printerr("After trim word embedding, " + str(word_embedding.embeddings.size(0)) + " words")
    word_embedding.extend_by_counts(snli.train_word_counts)
    printerr("After adding training words, " + str(word_embedding.embeddings.size(0)) + " words")

    # mark word ids in snli trees
    for _data in snli.train:
        _data['p_tree'].mark_word_id(word_embedding)
        _data['h_tree'].mark_word_id(word_embedding)
    for _data in snli.dev:
        _data['p_tree'].mark_word_id(word_embedding)
        _data['h_tree'].mark_word_id(word_embedding)

    config = {'hidden_dim': 400, 'relation_num': 3}
    root_align = RootAlign(word_embedding, config)
    optimizer = optim.Adadelta(root_align.parameters())

    for _data in snli.dev:
        p_tree = _data['p_tree']
        h_tree = _data['h_tree']
        output = root_align(p_tree, h_tree)
        max_v = output.data.max(1)[1][0][0]
        right = True if max_v == _data['label'] else False
        print 'label: ', _data['label'], '\tpred: ', max_v, '\tright: ', right

    for i in xrange(10):
        train_loss = 0
        for _data in snli.train:
            p_tree = _data['p_tree']
            h_tree = _data['h_tree']
            target = Variable(torch.LongTensor([_data['label']]))
            optimizer.zero_grad()
            output = root_align(p_tree, h_tree)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
        print 'loss:', train_loss

    for _data in snli.dev:
        p_tree = _data['p_tree']
        h_tree = _data['h_tree']
        output = root_align(p_tree, h_tree)
        max_v = output.data.max(1)[1][0][0]
        right = True if max_v == _data['label'] else False
        print 'label: ', _data['label'], '\tpred: ', max_v, '\tright: ', right

train()