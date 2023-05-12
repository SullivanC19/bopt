import numpy as np
from scipy.special import loggamma

from sklearn.tree._tree import Tree as SklearnTree

from typing import SupportsFloat, Union, FrozenSet

def log_beta(arr: np.ndarray) -> SupportsFloat:
    return np.sum(loggamma(arr)) - loggamma(np.sum(arr))

def label_counts(labels: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    return np.bincount(labels.astype(np.int64), minlength=int(np.max(label_values)) + 1)[label_values.astype(np.int64)]

def log_prob(label_prior: Union[np.ndarray, SupportsFloat], label_counts: np.ndarray) -> SupportsFloat:
    if np.sum(label_counts) == 0:
        return 0.0
    return log_beta(label_counts + label_prior) - log_beta(label_prior)

class Tree:
    def __init__(self, root=None):
        self.root = TreeNode() if root is None else root

    @classmethod
    def from_dict(cls, tree_dict: dict):
        return Tree(TreeNode.from_dict(tree_dict))
    
    @classmethod
    def from_sklearn_tree(cls, sklearn_tree: SklearnTree):
        return Tree(TreeNode.from_sklearn_node(0, sklearn_tree))

    def size(self):
        return self.root.size()
    
    def regions(self, X):
        return self.root.regions(frozenset(range(len(X))), X)

    def predict(self, X):
        pred = np.zeros(len(X), dtype=int)
        return self.root.predict(X)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def log_likelihood(self, X, y, alpha=2.0):
        return self.root.log_likelihood(X, y, np.unique(y), alpha=alpha)
    
    def __str__(self) -> str:
        return self.root.__str__()

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
    
    @classmethod
    def from_sklearn_node(cls, node_id: int, sklearn_tree: SklearnTree):
        node = TreeNode()
        children_left = sklearn_tree.children_left
        children_right = sklearn_tree.children_right
        feature = sklearn_tree.feature
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            node.feature = feature[node_id]

            # sklearn split is inverted for binary features
            node.left = TreeNode.from_sklearn_node(children_right[node_id], sklearn_tree)
            node.right = TreeNode.from_sklearn_node(children_left[node_id], sklearn_tree)
        else:
            node.value = np.argmax(sklearn_tree.value[node_id][0])
        return node


    def is_leaf(self):
        return self.left is None and self.right is None
    
    def size(self):
        if self.is_leaf():
            return 1
        return 1 + self.left.size() + self.right.size()
    
    def predict(self, x):
        if self.is_leaf():
            return np.full(len(x), fill_value=self.value, dtype=int)
        left = x[:, self.feature] == 1
        right = np.logical_not(left)
        pred = np.zeros(len(x), dtype=int)
        pred[left] = self.left.predict(x[left])
        pred[right] = self.right.predict(x[right])
        return pred
        
    def log_likelihood(self, X, y, label_values, alpha=2.0):
        if self.is_leaf():
            return log_prob(np.full(len(label_values), fill_value=alpha / len(label_values)), label_counts(y, label_values))
        else:
            left_indices = X[:, self.feature] == 1
            right_indices = np.logical_not(left_indices)
            return self.left.log_likelihood(X[left_indices], y[left_indices], label_values, alpha=alpha) + \
                     self.right.log_likelihood(X[right_indices], y[right_indices], label_values, alpha=alpha)
        
    def regions(self, region: FrozenSet[int], X: np.ndarray):
        if self.is_leaf():
            return [region]
        left_indices = frozenset(np.where(X[:, self.feature] == 1)[0])
        right_indices = frozenset(np.where(X[:, self.feature] == 0)[0])
        return self.left.regions(region.intersection(left_indices), X) + \
                self.right.regions(region.intersection(right_indices), X)

    def __str__(self) -> str:
        if self.is_leaf():
            return str(self.value)
        return f'({self.feature} {self.left} {self.right})'