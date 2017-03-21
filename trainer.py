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

args = get_args()
args.cuda = not args.gpu_id == -1 and torch.cuda.is_available()

if args.cuda:
    torch.cuda.manual_seed(2415)
    torch.cuda.set_device(args.gpu_id)


class Trainer(object):
    def __init__(self, verbose=True):
        self.verbose = verbose
        if self.verbose:
            printerr('Word embedding path: ' + args.word_embedding)
        self.word_embedding = WordEmbedding(args.word_embedding)

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
                  'cuda_flag': args.cuda, 'drop_p': args.drop_out}
        self.model = DoubleAttention_vRNN(self.word_embedding, config)
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
