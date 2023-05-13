import numpy as np
from sklearn.model_selection import train_test_split
from pydl85 import DL85Classifier, Cache_Type
import time

from math import lgamma, log

from typing import List

import pickle
from argparse import ArgumentParser
import os

F_DATASET = "datasets/{dataset}.txt"
DIR_TREES = "results/trees/{dataset}/samplesize={sample_size}_sample={idx}_seed={seed}/"
F_OPT_TREE = "opt_tree_dept=_{depth}.pkl"
F_SPARSE_OPT_TREE = "sparse_opt_tree_lambda={lamb}.pkl"
F_MAP_TREE = "map_tree_alpha={alpha}_alphas={alpha_s}_betas={beta_s}.pkl"

def load_k_fold_data(dataset: str, seed: int, k: int=10):
      data = np.genfromtxt(F_DATASET.format(dataset=dataset), delimiter=' ')
      X, y = data[:, 1:], data[:, 0]
      fold = np.random.RandomState(seed=seed).permutation(len(X)) % k
      n, d = X.shape
      l = np.max(y) + 1
      return [(X[fold == i], y[fold == i]) for i in range(k)], n, d, l

def load_data(dataset: str, sample_sizes: List[int], num_samples_taken: int, seed: int):
      data = np.genfromtxt(F_DATASET.format(dataset=dataset), delimiter=' ')
      X, y = data[:, 1:], data[:, 0]
      n, d = X.shape
      l = np.max(y) + 1
      if max(sample_sizes) >= n:
            return None, n, d, l
      else:
            rs = np.random.RandomState(seed=seed)
            return [[train_test_split(X, y, train_size=sample_size, random_state=rs) for _ in range(num_samples_taken)] for sample_size in sample_sizes], n, d, l

def run_experiment(
            dataset: str,
            alpha: float,
            alpha_s: float,
            beta_s: float,
            lamb: float,
            opt_depths: List[int],
            sample_sizes: List[int],
            num_samples_taken: int,
            seed: int,
            min_samples: int,
            max_features: int):
      print(dataset)

      print("Loading data...")
      samples, n, d, l = load_data(dataset, sample_sizes, num_samples_taken, seed)
      print("Data loaded")

      if n < min_samples:
            raise ValueError("Number of samples is too low, it is {}, but must be at least {}".format(n, min_samples))
      
      elif d > max_features:
            raise ValueError("Number of features is too high, it is {}, but must be at most {}".format(d, max_features))

      elif l > 2:
            raise ValueError("Can only handle binary classification")

      opt_trees = []
      sparse_opt_trees = []
      map_trees = []
      for i, sample_size in enumerate(sample_sizes):

            opt_trees.append([])
            sparse_opt_trees.append([])
            map_trees.append([])

            print("Sample size: {}".format(sample_size))
            for j, (X_train, _, y_train, _) in enumerate(samples[i]):
                  print("Sample: {}".format(j))
            
                  # DEFINE ERROR FUNCTIONS

                  lowest_possible_depth = len(X_train)
                  tiebreak_val = np.argmax(np.bincount(y_train.astype(np.int32)))
                  all_depths = np.arange(lowest_possible_depth + 1)
                  split_pen = np.log(alpha_s) - beta_s * np.log(1 + all_depths)
                  stop_pen = np.log(1 - alpha_s * np.power(1 + all_depths, -beta_s))
                  node_penalty = lamb * len(X_train)
                  alpha_prior_term = lgamma(alpha) - 2 * lgamma(alpha / 2)
                  
                  def error_opt(label_counts):
                        c0, c1 = label_counts
                        return min(c0, c1), \
                              tiebreak_val if c0 == c1 else (0 if c0 > c1 else 1), \
                              0.0

                  def error_sparse_opt(label_counts):
                        c0, c1 = label_counts
                        return min(c0, c1) + node_penalty, \
                              tiebreak_val if c0 == c1 else (0 if c0 > c1 else 1), \
                              node_penalty

                  def split_penalty_map(depth, splits):
                        return 0 if splits == 0 else -(-stop_pen[depth] + split_pen[depth] + 2 * stop_pen[depth + 1] - log(splits))

                  def error_map(label_counts):
                        c0, c1 = label_counts
                        leaf_error = -(lgamma(c0 + alpha / 2) + lgamma(c1 + alpha / 2) - lgamma(c0 + c1 + alpha) + alpha_prior_term)
                        perfect_split_error = -((lgamma(c0 + alpha / 2) + lgamma(alpha / 2) - lgamma(c0 + alpha)) + (lgamma(c1 + alpha / 2) + lgamma(alpha / 2) - lgamma(c1 + alpha)) + 2 * alpha_prior_term)
                        return leaf_error, tiebreak_val if c0 == c1 else (0 if c0 > c1 else 1), min(leaf_error, perfect_split_error)

                  opt_trees[i].append([])
                  for opt_max_depth in opt_depths:
                        clf_opt = DL85Classifier(
                              cache_type=Cache_Type.Cache_HashCover,
                              depth_two_special_algo=False,
                              similar_lb=False,
                              similar_for_branching=False,
                              fast_error_lb_function=error_opt,
                              max_depth=opt_max_depth)
                        start = time.perf_counter()
                        clf_opt.fit(X_train, y_train)
                        duration = time.perf_counter() - start
                        opt_trees[i][j].append((clf_opt.tree_, duration))

                  clf_sparse_opt = DL85Classifier(
                        cache_type=Cache_Type.Cache_HashCover,
                        depth_two_special_algo=False,
                        similar_lb=False,
                        similar_for_branching=False,
                        fast_error_lb_function=error_sparse_opt,
                        max_depth=lowest_possible_depth)
                  start = time.perf_counter()
                  clf_sparse_opt.fit(X_train, y_train)
                  duration = time.perf_counter() - start
                  sparse_opt_trees[i].append((clf_sparse_opt.tree_, duration))
                  
                  clf_map = DL85Classifier(
                        cache_type=Cache_Type.Cache_HashCover,
                        depth_two_special_algo=False,
                        similar_lb=False,
                        similar_for_branching=False,
                        split_penalty_function=split_penalty_map,
                        fast_error_lb_function=error_map,
                        max_depth=lowest_possible_depth)
                  start = time.perf_counter()
                  clf_map.fit(X_train, y_train)
                  duration = time.perf_counter() - start
                  map_trees[i].append((clf_map.tree_, duration))

      return opt_trees, sparse_opt_trees, map_trees


if __name__ == "__main__":
      parser = ArgumentParser()
      parser.add_argument('dataset', type=str)
      parser.add_argument('--alpha', type=float, default=5.0)
      parser.add_argument('--alpha_s', type=float, default=0.95)
      parser.add_argument('--beta_s', type=float, default=0.5)
      parser.add_argument('--lamb', type=float, default=0.005)
      parser.add_argument('--opt_depths', type=int, nargs='+', default=[2, 3, 4, 5])
      parser.add_argument('--seed', type=int, default=42)
      parser.add_argument('--sample_sizes', type=int, nargs='+', default=[10, 20, 40, 80, 160, 320, 640])
      parser.add_argument('--num_samples_taken', type=int, default=10)
      parser.add_argument('--min_samples', type=int, default=200)
      parser.add_argument('--max_features', type=int, default=50)
      args = parser.parse_args()

      opt_trees, sparse_opt_trees, map_trees = run_experiment(
            args.dataset,
            args.alpha,
            args.alpha_s,
            args.beta_s,
            args.lamb,
            args.opt_depths,
            args.sample_sizes,
            args.num_samples_taken,
            args.seed,
            args.min_samples,
            args.max_features)

      for i, sample_size in enumerate(args.sample_sizes):
            for j in range(args.num_samples_taken):
                  dir_tree = DIR_TREES.format(
                        dataset=args.dataset,
                        seed=args.seed,
                        idx=j,
                        sample_size=sample_size)
                  os.makedirs(dir_tree, exist_ok=True)

                  for d, opt_max_depth in enumerate(args.opt_depths):
                        opt_tree_path = os.path.join(dir_tree, F_OPT_TREE.format(depth=opt_max_depth))
                        pickle.dump(opt_trees[i][j][d], open(opt_tree_path, 'wb'))

                  sparse_opt_tree_path = os.path.join(dir_tree, F_SPARSE_OPT_TREE.format(lamb=args.lamb))
                  pickle.dump(sparse_opt_trees[i][j], open(sparse_opt_tree_path, 'wb'))

                  map_tree_path = os.path.join(dir_tree, F_MAP_TREE.format(alpha=args.alpha, alpha_s=args.alpha_s, beta_s=args.beta_s))
                  pickle.dump(map_trees[i][j], open(map_tree_path, 'wb'))
      


