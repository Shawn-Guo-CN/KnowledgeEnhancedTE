"""

  Loads word embeddings from text word2vec format. The loaded word embeddings are cached.

"""

import numpy as np
import os
from utils import *


class WordEmbedding(object):
    def __init__(self, path):
        self.max_word_width = 1024
        self.OOV_SYM = '<OOV>'

        cache_path = path + '.pickle'
        if not os.path.isfile(cache_path):
            printerr('Loading embedding from raw file...')
            self.vocab, self.embeddings = self.load_from_raw(path, cache_path)
        else:
            printerr('Loading embedding from cache file...')
            cache = torch.load(cache_path)
            self.vocab, self.embeddings = cache[0], cache[1]

        self.word2idx = None

        printerr('%d words loaded.' % len(self.vocab))

    def vocab_has_word(self, word):
        assert not self.vocab is None
        return word in self.vocab

    def get_word_idx(self, word):
        if self.word2idx is None:
            self.word2idx = {}
            index = 0
            for w in self.vocab:
                self.word2idx[w] = index
                index += 1

        return self.word2idx[word]

    def load_from_raw(self, path, cache_path):
        file = open(path, 'r')
        num_words, num_dim = file.readline().strip().split(' ')
        num_words = int(num_words)
        num_dim = int(num_dim)

        vocab = []
        embeddings = []
        for line in file:
            line = line.strip()
            word = line.split(' ')[0]
            vec = [float(x) for x in line.split(' ')[1:]]
            vocab.append(word)
            embeddings.append(vec)

        embeddings = torch.from_numpy(np.asarray(embeddings))

        printerr('Writing embeddings to ' + cache_path + '...')
        torch.save([vocab, embeddings], cache_path)

        return vocab, embeddings

    def save(self, path):
        f = open(path, 'w')
        num_words = len(self.vocab)
        dim = self.embeddings.size(1)
        print>> f, num_words, dim

        for i in xrange(num_words):
            print>> f, self.vocab[i], ' '.join([str(x) for x in self.embeddings[i]])
        f.close()

    def trim_by_counts(self, word_counts):
        # remove words w/o counts
        trimmed_vocab = []
        trimmed_vocab.append(self.OOV_SYM)

        for w in self.vocab:
            if word_counts.has_key(w):
                trimmed_vocab.append(w)

        trimmed_embeddings = torch.Tensor(len(trimmed_vocab), self.embeddings.size(1))

        for i in xrange(len(trimmed_vocab)):
            if trimmed_vocab[i] == self.OOV_SYM:
                trimmed_embeddings[i] = (torch.rand(self.embeddings.size(1)) - 0.5) / 10
            else:
                trimmed_embeddings[i] = self.embeddings[self.get_word_idx(trimmed_vocab[i])]

        self.vocab = trimmed_vocab
        self.embeddings = trimmed_embeddings
        self.word2idx = None

    def extend_by_counts(self, word_counts):
        extended_vocab = []
        for w in self.vocab:
            extended_vocab.append(w)

        extend_words = word_counts.keys()
        for w in extend_words:
            if not w in self.vocab:
                extended_vocab.append(w)

        extended_embeddings = torch.Tensor(len(extended_vocab), self.embeddings.size(1))
        for i in xrange(len(extended_vocab)):
            if extended_vocab[i] == self.OOV_SYM:
                extended_embeddings[i] = (torch.rand(self.embeddings.size(1)) - 0.5) / 10
            elif self.vocab_has_word(extended_vocab[i]):
                extended_embeddings[i] = self.embeddings[self.get_word_idx(extended_vocab[i])]
            else:
                extended_embeddings[i] = (torch.rand(self.embeddings.size(1)) - 0.5) / 10

        self.vocab = extended_vocab
        self.embeddings = extended_embeddings
        self.word2idx = None

    def convert(self, words):
        # converts the words to a vector of indices of word embeddings
        indices = torch.IntTensor(len(words))
        for w in words:
            idx = self.get_word_idx(w)
            if idx is None:
                idx = self.get_word_idx(self.OOV_SYM)
            indices[i] = idx

        return indices
