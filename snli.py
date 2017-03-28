"""

    Loads SNLI entailment dataset.

"""

from tree import Tree
from utils import *
import os
import cPickle


class SNLI(object):
    def __init__(self, snli_path_prefix=None, train_size=0,
                 lower_case=True, verbose=True):
        self.num_relations = 3
        self.relations = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        self.rev_relations = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}

        self.train_size = train_size
        self.lower_case = lower_case
        self.verbose = verbose

        self.train_word_counts = {}
        self.dev_word_counts = {}
        self.word_counts = {}

        self.train_phrase_counts = {}
        self.dev_phrase_counts = {}
        self.phrase_counts = {}

        self.load_from_cache = False

        def _load_from_raw(train_name, dev_name, test_name):
            printerr('loading snli dataset from ' + snli_path_prefix + train_name +
                     ' and ' + snli_path_prefix + dev_name +
                     ' and ' + snli_path_prefix + test_name)
            # load train data
            self.train = self.load_data_file(snli_path_prefix + train_name, self.train_word_counts,
                                             self.train_phrase_counts)
            words = self.train_word_counts.keys()
            for w in words:
                self.dev_word_counts[w] = self.train_word_counts[w]
            phrases = self.train_phrase_counts.keys()
            for p in phrases:
                self.dev_phrase_counts[p] = self.train_phrase_counts[p]

            # load dev data
            self.dev = self.load_data_file(snli_path_prefix + dev_name, self.dev_word_counts,
                                           self.dev_phrase_counts)
            words = self.dev_word_counts.keys()
            for w in words:
                self.word_counts[w] = self.dev_word_counts[w]
            phrases = self.dev_phrase_counts.keys()
            for p in phrases:
                self.phrase_counts[p] = self.dev_phrase_counts[p]

            # load test data
            self.test = self.load_data_file(snli_path_prefix + test_name, self.word_counts,
                                            self.phrase_counts)

            if self.train_size > 0:
                self.train = self.train[:self.train_size]

            if self.verbose:
                printerr('SNLI train: %d pairs' % len(self.train))
                printerr('SNLI dev: %d pairs' % len(self.dev))
                printerr('SNLI test: %d pairs' % len(self.test))

        if snli_path_prefix is not None:
            if os.path.exists(snli_path_prefix + 'train.text') \
                    and os.path.exists(snli_path_prefix + 'dev.text') \
                    and os.path.exists(snli_path_prefix + 'test.text'):
                self.load_from_cache = True
                _load_from_raw('train.text', 'dev.text', 'test.text')
            else:
                _load_from_raw('train.txt', 'dev.txt', 'test.txt')

    def inc_word_counts(self, word, counter):
        if counter.has_key(word):
            counter[word] += 1
        else:
            counter[word] = 1

    def inc_phrase_counts(self, phrase, counter):
        if counter.has_key(phrase):
            counter[phrase] += 1
        else:
            counter[phrase] = 1

    def load_data_file(self, file_path, word_counter, phrase_counter):
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

                p_tree_str = ' '.join(premise)
                h_tree_str = ' '.join(hypothese)

                p_tree = Tree()
                h_tree = Tree()
                p_tree.parse(p_tree_str)
                h_tree.parse(h_tree_str)

                for p in p_tree.get_sentence():
                    self.inc_word_counts(p, word_counter)
                    p_tree.sent = None
                for h in h_tree.get_sentence():
                    self.inc_word_counts(h, word_counter)
                    h_tree.sent = None

                if self.load_from_cache:
                    for p in p_tree.get_phrases():
                        self.inc_word_counts(p, phrase_counter)
                    for h in h_tree.get_phrases():
                        self.inc_phrase_counts(h, phrase_counter)

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
