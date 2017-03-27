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
        self.word_embedding.trim_by_counts(self.data.word_counts, self.data.phrase_counts)
        if self.verbose:
            printerr("After trim word embedding, " + str(self.word_embedding.embeddings.size(0)) + " words")
        self.word_embedding.extend_by_counts(self.data.word_counts)
        if self.verbose:
            printerr("After adding training words, " + str(self.word_embedding.embeddings.size(0)) + " words")

        def _prune_and_mark_id(data_set):
            for data in data_set:
                data['p_tree'].prune(self.word_embedding)
                data['p_tree'].mark_word_id(self.word_embedding)
                data['h_tree'].prune(self.word_embedding)
                data['h_tree'].mark_word_id(self.word_embedding)

        # mark word ids in snli trees
        _prune_and_mark_id(self.data.train)
        _prune_and_mark_id(self.data.dev)
        _prune_and_mark_id(self.data.test)

        config = {'hidden_dim': args.hidden_dim, 'relation_num': 3,
                  'cuda_flag': args.cuda, 'drop_p': args.drop_out}
        self.model = AttentionFromH2P_vRNN(self.word_embedding, config)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=args.learning_rate)
        if not args.dump is None:
            self.dump = args.dump + self.model.name
        else:
            self.dump = None

        if args.load_params is not None:
            printerr('loading params from' + args.load_params)
            loaded = torch.load(args.load_params)
            self.model.load_state_dict(loaded)

        if args.cuda:
            self.model.cuda()
            printerr('Using GPU %s' % str(args.gpu_id))
        printerr('Training ' + self.model.name)

    def train(self):
        best_dev_acc = 0.0
        best_test_acc = 0.0
        profiler = SimpleProfiler()

        for i in xrange(0, args.epoches):
            if self.verbose:
                printerr("Starting epoch%d" % i)

            profiler.reset('train')
            profiler.start("train")
            train_loss = self.train_step(self.data.train)
            profiler.pause("train")

            print 'model:', self.model.name, 'epoch:', i, 'train_loss:', train_loss, \
                'time:', profiler.get_time('train')

            if (i + 1) % args.valid_freq == 0:
                profiler.reset('dev')
                profiler.start('dev')
                dev_acc = self.eval_step(self.data.dev)
                profiler.pause('dev')
                print '\t evaluating at epoch:', i, 'acc:', dev_acc, 'time:', profiler.get_time('dev')

                if best_dev_acc < dev_acc:
                    best_dev_acc = dev_acc

                    if self.dump is not None:
                        file_name = "%s.epoch%d.acc%.4f.pickle" % (self.dump, i, dev_acc)
                        printerr("saving weights to " + file_name)
                        torch.save(self.model.state_dict(), file_name)

                profiler.reset('test')
                profiler.start('test')
                test_acc = self.eval_step(self.data.test)
                profiler.pause('test')
                print '\t evaluating at epoch:', i, 'test acc:', test_acc, 'time:', profiler.get_time('dev')

                if best_test_acc < test_acc:
                    best_test_acc = test_acc

                    if self.dump is not None:
                        file_name = "%s.epoch%d.testacc%.4f.pickle" % (self.dump, i, test_acc)
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
        self.model.set_train_flag(False)
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

        self.model.set_train_flag(True)


t = Trainer()

if not args.eval is None:
    printerr("loading weights from " + args.eval)
    loaded = torch.load(args.eval)
    t.model.load_state_dict(loaded)
    eval_acc = t.eval_step(t.data.dev)
    printerr("dev acc %f" % eval_acc)
else:
    t.train()
