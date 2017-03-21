"""

    Loads SNLI entailment dataset.

"""

from tree import Tree
from utils import *
import os
import cPickle

class SNLI(object):
    def __init__(self, snli_path_prefix=None, train_size=0, lower_case=True, verbose=True):
        self.num_relations = 3
        self.relations = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        self.rev_relations = []
        for r in self.relations:
            self.rev_relations.append(r)
        self.train_size = train_size
        self.lower_case = lower_case
        self.verbose = verbose

        self.train_word_counts = {}
        self.dev_word_counts = {}
        self.word_counts = {}

        if not snli_path_prefix is None:
            printerr('loading snli dataset from ' + snli_path_prefix + 'train.txt' +
                     ' and ' + snli_path_prefix + 'dev.txt' +
                     ' and ' + snli_path_prefix + 'test.txt')
            self.train = self.load_data_file(snli_path_prefix + 'train.txt', self.train_word_counts)
            words = self.train_word_counts.keys()
            for w in words:
                self.dev_word_counts[w] = self.train_word_counts[w]
            self.dev = self.load_data_file(snli_path_prefix + 'dev.txt', self.dev_word_counts)
            words = self.dev_word_counts.keys()
            for w in words:
                self.word_counts[w] = self.dev_word_counts[w]
            self.test = self.load_data_file(snli_path_prefix + 'test.txt', self.word_counts)


            if self.train_size > 0:
                self.train = self.train[:self.train_size]

            if self.verbose:
                printerr('SNLI train: %d pairs' % len(self.train))
                printerr('SNLI dev: %d pairs' % len(self.dev))

    def inc_word_counts(self, word, counter):
        if counter.has_key(word):
            counter[word] += 1
        else:
            counter[word] = 1

    def load_data_file(self, file_path, word_counter):
        data = []
        f = open(file_path, 'r')
        for line in f:
            line_split = line.strip().split('\t')
            gold_label = line_split[0]
            if self.relations.has_key(gold_label):
                premise = line_split[1].strip().split()
                hypothese = line_split[2].strip().split()

                if self.lower_case:
                    premise = [p.lower() for p in premise]
                    hypothese = [h.lower() for h in hypothese]

                for p in premise:
                    self.inc_word_counts(p, word_counter)
                for h in hypothese:
                    self.inc_word_counts(h, word_counter)

                p_tree_str = ' '.join(premise)
                h_tree_str = ' '.join(hypothese)

                p_tree = Tree()
                h_tree = Tree()
                p_tree.parse(p_tree_str)
                h_tree.parse(h_tree_str)

                data.append({'label': self.relations[gold_label],
                             'id': len(data),
                             # 'premise': ptree_str,
                             'p_tree': p_tree,
                             # 'hypothese': htree_str,
                             'h_tree': h_tree})
            else:
                # printerr('Error loading' + line)
                pass

        return data
