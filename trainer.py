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
from model_entailment import StructuredEntailmentModel

torch.manual_seed(123)

parser = argparse.ArgumentParser(description='Structured Attention PyTorch Version')
parser.add_argument('-t', '--train-size', type=int, default=0,
                    help='Number of samples used in training (default: 0)')
parser.add_argument('--dim', type=int, default=150,
                    help='LSTM memory dimension')
parser.add_argument('-e', '--epoches', type=int, default=30,
                    help='Number of training epoches')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--hiddenrel', type=int, default=150,
                    help='Number of hidden relations')
parser.add_argument('--dataset_prefix', type=str, default='./sampledata',
                    help='Prefix of path to dataset')
parser.add_argument('-d', '--drop_out', type=float, default=0.2,
                    help='Dropout rate')
parser.add_argument('-w', '--word_embedding', type=str, default='./sampledata/wordembedding',
                    help='Path to word embedding')
parser.add_argument('--gpu_id', type=int, default=None,
                    help='The gpu device to use. None means use only CPU.')
parser.add_argument('--interactive', type=bool, default=True,
                    help='Show progress interactively')
parser.add_argument('--dump', default=None, help='Weights dump')
parser.add_argument('--eval', default=None, help='Evaluate weights')
parser.add_argument('--oovonly', type=bool, default=True,
                    help='Update OOV embeddings only')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.cuda_device)


class Trainer(object):
    def __init__(self, verbose):
        self.verbose = verbose or True
        if self.verbose:
            printerr('Word embedding path: ' + args.word_embedding)
        self.word_embedding = WordEmbedding(args.word_embedding)

        if self.verbose:
            printerr('Dataset prefix:' + args.dataset_prefix)

        self.dump = args.dump

        self.data = SNLI(args.dataset_prefix, args.train_size, True, True)

        # trim the word embeddings to contain only words in the dataset
        if self.verbose:
            printerr("Before trim word embedding, " + str(self.word_embedding.embeddings.size(0)) + " words")
        print self.data.word_counts
        self.word_embedding.trim_by_counts(self.data.word_counts)
        words_from_embedding = self.word_embedding.embeddings.size(0)
        if self.verbose:
            printerr("After trim word embedding, " + str(words_from_embedding) + " words")

        self.word_embedding.extend_by_counts(self.data.train_word_counts)

        if self.verbose:
            printerr("After adding training words, " + str(self.word_embedding.embeddings.size(0)) + " words")

        self.model = StructuredEntailmentModel({'word_emb': self.word_embedding,
                                                'repr_dim': args.dim,
                                                'num_relations': self.data.num_relations,
                                                'learning_rate': args.learning_rate,
                                                'batch_size': args.batch_size,
                                                'dropout': args.dropout,
                                                'interactive': True,
                                                'words_from_embbedding': words_from_embedding,
                                                'update_oov_only': args.oovonly,
                                                'hiddenrel ': args.hiddenrel,
                                                'dataset': self.data,
                                                'verbose': self.verbose})

    def train(self):
        best_train_acc, best_dev_acc = 0.0, 0.0
        train = self.data.train

        profiler = SimpleProfiler()

        for i in xrange(1, args.epoches):
            if self.verbose:
                printerr("Starting epoch%d" % i)

            profiler.reset()
            profiler.start("train")
            train_info = self.model.train(train)
            profiler.pause("train")

            profiler.start("dev")
            dev_info = self.model.evaluate(self.data.dev)
            profiler.pause("dev")

            best_train_suffix, best_dev_suffix = "", ""
            if best_train_acc < train_info['acc']:
                best_train_acc = train_info['acc']
                best_train_suffix = '+'

            if best_dev_acc < dev_info['acc']:
                best_dev_acc = dev_info['acc']
                best_dev_suffix = "+"

            printerr("At epoch %d, train %.2fs loss %f acc %f%s dev %.2fs acc %f%s" % (
                i, profiler.get_time('train'), train_info['loss'], train_info['acc'], best_train_suffix,
                profiler.get_time('dev'), dev_info['acc'], best_dev_suffix))

            if not self.dump is None:
                file_name = "%s.%d.pickle" % (self.dump, i)
                printerr("saving weights to " + file_name)
                torch.save(self.model.params, file_name)


t = Trainer()

if not args.eval is None:
    printerr("loading weights from " + args.eval)
    loaded = torch.load(args.eval)
    printerr("loaded params size: " + get_tensor_size(loaded))
    t.model.params.copy_(loaded)
    eval_info = t.model.evaluate(t.data.dev)
    printerr("dev acc %f" % eval_info["acc"])
else:
    t.train()
