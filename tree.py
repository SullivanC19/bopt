import numpy as np
from scipy.special import loggamma

from typing import SupportsFloat

def log_beta(arr: np.ndarray) -> SupportsFloat:
    return np.sum(loggamma(arr)) - loggamma(np.sum(arr))

def label_counts(labels: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    return np.bincount(labels, minlength=int(np.max(label_values)) + 1)[label_values]

def log_prob(label_priors: np.ndarray, label_counts: np.ndarray) -> SupportsFloat:
    if np.sum(label_counts) == 0:
        return float('-inf')
    return log_beta(label_counts + label_priors) - log_beta(label_priors)

class Tree:
    def __init__(self, root=None):
        self.root = TreeNode() if root is None else root

    def log_likelihood(self, X, y):
        # get from previous project
        # add log prior
        # add stability comparison with other tree
        pass

class TreeNode:
    def __init__(self, feature: int=None, value: int=None, left: 'TreeNode'=None, right: 'TreeNode'=None):
        self.left = left
        self.right = right
        self.value = value

    @classmethod
    def from_dict(cls, tree_dict: dict):
        node = TreeNode()
        if 'value' in tree_dict: # leaf node
            node.value = tree_dict['value']
        else: # internal node
            node.feature = tree_dict['feature']
            node.left = TreeNode.from_dict(tree_dict['left'])
            node.right = TreeNode.from_dict(tree_dict['right'])
        return node

    def is_leaf(self):
        return self.left is None and self.right is None