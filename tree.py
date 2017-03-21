"""

  Tree structure.

"""
import torch
from utils import is_stop_word


class Tree(object):
    def __init__(self, val=None, children=None):
        # val contains the word characters, if val is not None, it means this node is a leaf node
        self.val = val
        self.children = children

        self.leaf_id = None
        self.postorder_id = None
        # word_id is the idx in embedding matrix
        self.word_id = None

        # sent is a list of the words
        self.sent = None

        # calculate_result contains the calculate result of tree_models, and it can be either a
        # Variable or a list of Variables(depends on the computation methods, LSTM or Vanilla RNN)
        self.calculate_result = None
        # gather result is to concatenate the calculate_result of every node
        self.gather_result = None
        # this is used for saving attention calculate result temporarily
        # attn should be matrix of float number, and shape is 1xn
        self.attn = None
        # attn_result is
        self.attn_result = None

        self.related_terms = None

    def __str__(self):
        if self.val is None:
            return '( ' + ' '.join([str(c) for c in self.children]) + ' )'
        else:
            return self.val

    def parse(self, tree_str, prune_last_period=True):
        """
            Loads a tree from the input string.
            :param prune_last_period: whether last period should be pruned.
            :param tree_str: tree string in parentheses form.
        """
        self.build_from_str(tree_str, 0)
        if prune_last_period:
            self.prune_last_period()
        self.mark_leaf_id()
        self.mark_postorder()

    def build_from_str(self, tree_str, index):
        assert tree_str[index] == '('

        index += 1
        children = []
        while not tree_str[index] == ')':
            if tree_str[index] == '(':
                sub_t = Tree()
                index, _ = sub_t.build_from_str(tree_str, index)
                children.append(sub_t)
            else:
                r_pos = min(tree_str.index(' ', index), tree_str.index(')', index))
                leaf_word = tree_str[index: r_pos]
                if not leaf_word == '':
                    leaf = Tree(leaf_word, [])
                    children.append(leaf)
                index = r_pos + 1

            if tree_str[index] == ' ':
                index += 1

        assert tree_str[index] == ')'

        self.children = children
        return index + 1, children

    def mark_leaf_id(self):
        global tree_leaf_count
        tree_leaf_count = 0

        def func(node):
            global tree_leaf_count
            if not node.val is None:
                node.leaf_id = tree_leaf_count
                tree_leaf_count += 1

        self.inorder_traverse(func)
        tree_leaf_count = 0

    def mark_word_id(self, word_embedding):
        """
        :type word_embedding: an instance of WordEmbedding
        """
        def func(node):
            if not node.val is None:
                node.word_id = word_embedding.get_word_idx(node.val)

        self.postorder_traverse(func)

    def get_leaf_word_ids(self):
        ids = []
        def func(node):
            if not node.val is None:
                ids.append(node.word_id)
        self.postorder_traverse(func)

        return ids

    def print_leaf_word(self):
        def func(node):
            if not node.val is None:
                print node.val

        self.inorder_traverse(func)

    def get_sentence(self):
        if self.sent is None:
            global sent
            sent = []

            def func(node):
                if not node.val is None:
                    sent.append(node.val)

            self.postorder_traverse(func)

            self.sent = sent
            return self.sent
        else:
            return self.sent

    def get_str4conceptnet(self):
        sent = self.get_sentence()
        ret = []
        for w in sent:
            if not is_stop_word(w):
                ret.append(w)
        return '_'.join(ret)

    def mark_postorder(self):
        global tree_post_count
        tree_post_count = 0

        def func(node):
            global tree_post_count
            node.postorder_id = tree_post_count
            tree_post_count += 1

        self.postorder_traverse(func)
        tree_post_count = 0

    def prune_last_period(self):
        if self.val is None:
            assert len(self.children) == 2
            if self.children[1].val == '.':
                self.val = self.children[0].val
                self.children = self.children[0].children
            else:
                self.children[1].prune_last_period()

    # gather calculate_result of all nodes into a matrix
    def gather_calculate_result(self):
        if not hasattr(self, 'gather_result') or self.gather_result is None:
            global result
            result = None
            def _func(node):
                global result
                if result is None:
                    result = node.calculate_result
                else:
                    result = torch.cat((result, node.calculate_result))
            self.postorder_traverse(_func)
            self.gather_result = result

        return self.gather_result

    # calculate the attention representation of every node
    def get_attention_representation(self):
        if not hasattr(self, 'attn_result') or self.attn_result is None:
            def _func(node):
                if node.val is not None:
                    node.attn_result = node.attn
                else:
                    assert len(node.children) == 2
                    node.attn_result = node.children[0].attn_result + node.children[1].attn_result
            self.postorder_traverse(_func)
        return self.attn_result

    def prune(self):
        pass
        """
        function Tree:prune(test_func)
            -- return true is this tree node needs to be pruned
            if self.val == nil then
                -- internal node
                local leftprune = self.children[1]:prune(test_func)
                local rightprune = self.children[2]:prune(test_func)
                if leftprune == nil and rightprune == nil then
                  -- both left and right are pruned
                  return nil
                elseif leftprune == nil then return rightprune
                elseif rightprune == nil  then return leftprune
                else
                  self.children[1] = leftprune
                  self.children[2] = rightprune
                  return self
                end
            elseif test_func(self.val) then
                -- leaf node
                return nil
            else
                return self
            end
        end
        """

    def clear_vars(self):
        def func(node):
            if hasattr(node, 'calculate_result'):
                del node.calculate_result
            if hasattr(node, 'attn'):
                del node.attn
            if hasattr(node, 'attn_result'):
                del node.attn_result
            if hasattr(node, 'gather_result'):
                del node.gather_result
        self.postorder_traverse(func)

    def preorder_traverse(self, func):
        func(self)
        # assert len(self.children) == 2
        for c in self.children:
            c.preorder_traverse(func)

    def inorder_traverse(self, func):

        if self.val is None:
            assert len(self.children) == 2
            self.children[0].inorder_traverse(func)
            func(self)
            self.children[1].inorder_traverse(func)
        else:
            func(self)

    def postorder_traverse(self, func):
        # assert len(self.children) == 2
        for c in self.children:
            c.postorder_traverse(func)
        func(self)
