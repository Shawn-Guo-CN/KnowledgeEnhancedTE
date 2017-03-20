import requests


class ConceptNet(object):
    def __init__(self, word_embedding):
        self.word_embedding = word_embedding

    def query_related_terms(self, word, use_related=False):
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

    def query_node(self, node):
        if node.related_terms is not None:
            pass
        else:
            word = node.get_str4conceptnet()
            node.related_terms = self.query_related_terms(word)
