"""
  The trainer.
"""

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
from align_model import *

torch.manual_seed(1234)

parser = argparse.ArgumentParser(description='Structured Attention PyTorch Version')
parser.add_argument('-t', '--train-size', type=int, default=0,
                    help='Number of samples used in training (default: 0)')
parser.add_argument('--dim', type=int, default=150,
                    help='LSTM memory dimension')
parser.add_argument('-e', '--epoches', type=int, default=10,
                    help='Number of training epoches')
parser.add_argument('-lr', '--learning_rate', type=float, default=1.0,
                    help='Learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--hidden-dim', type=int, default=200,
                    help='Number of hidden units')
parser.add_argument('--dataset_prefix', type=str, default='./sampledata/',
                    help='Prefix of path to dataset')
parser.add_argument('-d', '--drop_out', type=float, default=0.2,
                    help='Dropout rate')
parser.add_argument('-w', '--word-embedding', type=str, default='./sampledata/wordembedding',
                    help='Path to word embedding')
parser.add_argument('--gpu-id', type=int, default=0,
                    help='The gpu device to use. None means use only CPU.')
parser.add_argument('--interactive', type=bool, default=True,
                    help='Show progress interactively')
parser.add_argument('--dump', default=None, help='Weights dump')
parser.add_argument('--eval', default=None, help='Evaluate weights')
parser.add_argument('--oovonly', type=bool, default=True,
                    help='Update OOV embeddings only')
parser.add_argument('-vfq', '--valid-freq', type=int, default=5,
                    help='Frequency of Validating model')

args = parser.parse_args()
args.cuda = not args.gpu_id is None and torch.cuda.is_available()

if args.cuda:
    torch.cuda.manual_seed(2415)
    torch.cuda.set_device(args.gpu_id)


class Trainer(object):
    def __init__(self, verbose=True):
        self.verbose = verbose
        if self.verbose:
            printerr('Word embedding path: ' + args.word_embedding)
        self.word_embedding = WordEmbedding(args.word_embedding)
        # word_embedding = WordEmbedding('./sampledata/wordembedding')
        if self.verbose:
            printerr('Dataset prefix:' + args.dataset_prefix)
        self.data = SNLI(args.dataset_prefix, args.train_size, True, True)

        # trim the word embeddings to contain only words in the dataset
        if self.verbose:
            printerr("Before trim word embedding, " + str(self.word_embedding.embeddings.size(0)) + " words")
        self.word_embedding.trim_by_counts(self.data.word_counts)
        if self.verbose:
            printerr("After trim word embedding, " + str(self.word_embedding.embeddings.size(0)) + " words")
        self.word_embedding.extend_by_counts(self.data.word_counts)
        if self.verbose:
            printerr("After adding training words, " + str(self.word_embedding.embeddings.size(0)) + " words")

        # mark word ids in snli trees
        for _data in self.data.train:
            _data['p_tree'].mark_word_id(self.word_embedding)
            _data['h_tree'].mark_word_id(self.word_embedding)
        for _data in self.data.dev:
            _data['p_tree'].mark_word_id(self.word_embedding)
            _data['h_tree'].mark_word_id(self.word_embedding)

        config = {'hidden_dim': args.hidden_dim, 'relation_num': 3,
                  'cuda_flag': args.cuda}
        self.model = RootAlign_BLSTM(self.word_embedding, config)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=args.learning_rate)
        if not args.dump is None:
            self.dump = args.dump + self.model.name
        else:
            self.dump = None

        if args.cuda:
            self.model.cuda()
            printerr('Using GPU %s' % str(args.gpu_id))
        printerr('Training ' + self.model.name)

    def train(self):
        best_dev_acc = 0.0
        profiler = SimpleProfiler()

        for i in xrange(0, args.epoches):
            if self.verbose:
                printerr("Starting epoch%d" % i)

            profiler.reset('train')
            profiler.start("train")
            train_loss = self.train_step(self.data.train)
            profiler.pause("train")

            print 'epoch:', i, 'train_loss:', train_loss, 'time:', profiler.get_time('train')

            if (i + 1) % args.valid_freq == 0:
                profiler.reset('dev')
                profiler.start('dev')
                dev_acc = self.eval_step(self.data.dev)
                profiler.pause('dev')
                print '\t evaluating at epoch:', i, 'acc:', dev_acc, 'time:', profiler.get_time('dev')

                if best_dev_acc < dev_acc:
                    best_dev_acc = dev_acc

                    if not self.dump is None:
                        file_name = "%s.epoch%d.acc%.4f.pickle" % (self.dump, i, dev_acc)
                        printerr("saving weights to " + file_name)
                        torch.save(self.model.state_dict(), file_name)

    def train_step(self, data):
        total = str(len(data))
        index = 0
        train_loss = 0.0
        print 'training model'
        for _data in data:
            p_tree = _data['p_tree']
            h_tree = _data['h_tree']
            if args.cuda:
                target = Variable(torch.LongTensor([_data['label']]).cuda())
            else:
                target = Variable(torch.LongTensor([_data['label']]))
            self.optimizer.zero_grad()
            output = self.model(p_tree, h_tree)
            p_tree.clear_vars()
            h_tree.clear_vars()
            # loss = F.nll_loss(output, target)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            index += 1
            train_loss += loss.data[0]
            print '\r', str(index), '/', total, 'loss:', loss.data[0],
        print '\t'
        return train_loss

    def eval_step(self, data):
        right_count = 0
        for _data in data:
            p_tree = _data['p_tree']
            h_tree = _data['h_tree']
            output = self.model(p_tree, h_tree)
            p_tree.clear_vars()
            h_tree.clear_vars()
            max_v = output.data.max(1)[1][0][0]
            right = True if max_v == _data['label'] else False
            if right:
                right_count += 1
        return float(right_count) / float(len(data))


t = Trainer()

if not args.eval is None:
    printerr("loading weights from " + args.eval)
    loaded = torch.load(args.eval)
    t.model.load_state_dict(loaded)
    eval_acc = t.eval_step(t.data.dev)
    printerr("dev acc %f" % eval_acc)
else:
    t.train()
