import requests
from snli import SNLI
from wordembedding import WordEmbedding
from utils import printerr
import cPickle

def query_related_terms(word, use_related=False):
    ret = set()

    query_prefix = 'http://api.conceptnet.io/query?node=/c/en/'
    obj = requests.get(query_prefix + word).json()
    for rel in obj['edges']:
        if rel['end'].has_key('term') and rel['end'].has_key('language') \
                and rel['end']['language'] == 'en':
            ret.add(rel['end']['label'])
        if rel['start'].has_key('term') and rel['start'].has_key('language') \
                and rel['start']['language'] == 'en':
            ret.add(rel['start']['label'])

    if use_related:
        relation_prefix = 'http://api.conceptnet.io/related/c/en/'
        relation_postfix = '?filter=/c/en'
        obj = requests.get(relation_prefix + word + relation_postfix).json()
        for rel in obj['related']:
            ret.add(rel['@id'].replace('_', ' ').split('/')[-1])

    if word in ret:
        ret.remove(word)

    return ret

def tree2set(t):
    global _set
    _set = set()
    def func(node):
        _set.add(node.get_str4conceptnet())
    t.postorder_traverse(func)
    return _set

word_embedding = WordEmbedding('./data/wordembedding')
snli = SNLI('./data/')

printerr("Before trim word embedding, " + str(word_embedding.embeddings.size(0)) + " words")
word_embedding.trim_by_counts(snli.word_counts)
printerr("After trim word embedding, " + str(word_embedding.embeddings.size(0)) + " words")
word_embedding.extend_by_counts(snli.train_word_counts)
printerr("After adding training words, " + str(word_embedding.embeddings.size(0)) + " words")

phrases = set()
print 'Gathering phrases in train data...'
for data in snli.train:
    phrases = phrases | tree2set(data['p_tree'])
    phrases = phrases | tree2set(data['h_tree'])
print 'done'
printerr('Gathering phrases in dev data...')
for data in snli.dev:
    phrases = phrases | tree2set(data['p_tree'])
    phrases = phrases | tree2set(data['h_tree'])
print 'done'
print 'Gathering phrases in test data...'
for data in snli.test:
    phrases = phrases | tree2set(data['p_tree'])
    phrases = phrases | tree2set(data['h_tree'])
print 'done'
print 'total num of phrases:', len(phrases)

related_terms = {}
idx = 0
for phrase in phrases:
    related_terms[phrase] = query_related_terms(phrase)
    idx += 1
    print '\rquerying', str(idx)+'/'+str(len(phrases)),
print ' '

with open('./data/dict_concept_related_terms.pickle', 'wb') as f:
    print 'saving dict to' + './data/dict_concept_related_terms.pickle'
    cPickle.dump(related_terms, f)
