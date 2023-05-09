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

    @classmethod
    def from_dict(cls, tree_dict: dict):
        return Tree(TreeNode.from_dict(tree_dict))

    def predict(self, X):
        pred = np.zeros(len(X), dtype=int)
        for i, x in enumerate(X):
            pred[i] = self.root.predict(x)
        return pred

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

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
        self.feature = feature

    @classmethod
    def from_dict(cls, tree_dict: dict):
        node = TreeNode()
        if 'value' in tree_dict: # leaf node
            node.value = tree_dict['value']
        else: # internal node
            node.feature = tree_dict['feat']
            node.left = TreeNode.from_dict(tree_dict['left'])
            node.right = TreeNode.from_dict(tree_dict['right'])
        return node

    def is_leaf(self):
        return self.left is None and self.right is None
    
    def predict(self, x):
        if self.is_leaf():
            return self.value
        if x[self.feature] == 1:
            return self.left.predict(x)
        else:
            return self.right.predict(x)