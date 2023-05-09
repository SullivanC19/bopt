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

DIR_TREES = "trees/{dataset}/k={k}_timelimit={time_limit}_seed={seed}/fold={fold}"
F_OPT_TREE = "opt_tree_dept=_{depth}.pkl"
F_SPARSE_OPT_TREE = "sparse_opt_tree_lambda={lamb}.pkl"
F_MAP_TREE = "map_tree_alpha={alpha}_alphas={alpha_s}_betas={beta_s}.pkl"

def load_data(dataset, k=10, seed=42):
      data = np.genfromtxt(F_DATASET.format(dataset=dataset), delimiter=' ')
      X, y = data[:, 1:], data[:, 0]
      fold = np.random.RandomState(seed=seed).permutation(len(X)) % k
      n, d = X.shape
      l = np.max(y) + 1
      return [(X[fold == i], y[fold == i]) for i in range(k)], n, d, l

def run_experiment(
            dataset: str,
            alpha: float,
            alpha_s: float,
            beta_s: float,
            lamb: float,
            opt_depths: List[int],
            time_limit: int,
            k: int,
            seed: int,
            cache_size_threshold: float,
            max_features: int):
      folds, n, d, l = load_data(dataset, k=k, seed=seed)

      cache_size_ub = min((n / k)  * log(2), d * log(3)) + log(n / k)
      if cache_size_ub > cache_size_threshold:
            raise ValueError("Cache size upper bound is too high, it is {}, but must be less than {}".format(cache_size_ub, cache_size_threshold))
      
      if d > max_features:
            raise ValueError("Number of features is too high, it is {}, but must be less than {}".format(d, max_features))

      if l > 2:
            raise ValueError("Number of classes is too high, it is {}, but must be less than 2".format(l))

      print(dataset)

      opt_trees = []
      sparse_opt_trees = []
      map_trees = []
      for i in range(k):
            X_train, y_train = folds[i]
            
            # DEFINE ERROR FUNCTIONS

            fold_size = len(X_train)
            lowest_possible_depth = len(X_train) + 1
            tiebreak_val = np.argmax(np.bincount(y_train.astype(np.int32)))
            all_depths = np.arange(lowest_possible_depth + 1)
            split_pen = np.log(alpha_s) - beta_s * np.log(1 + all_depths)
            stop_pen = np.log(1 - alpha_s * np.power(1 + all_depths, -beta_s))
            node_penalty = lamb * fold_size

            def split_penalty_map(depth):
                  return -(-stop_pen[depth] + split_pen[depth] + 2 * stop_pen[depth + 1])
            
            def error_opt(label_counts):
                  i, j = label_counts
                  return min(i, j), \
                        tiebreak_val if i == j else (0 if i > j else 1), \
                        0.0

            def error_sparse_opt(label_counts):
                  i, j = label_counts
                  return min(i, j) + node_penalty, \
                        tiebreak_val if i == j else (0 if i > j else 1), \
                        node_penalty

            def error_map(label_counts):
                  i, j = label_counts
                  return -((lgamma(i + alpha / 2) + lgamma(j + alpha / 2)) - lgamma(i + j + alpha)), \
                              tiebreak_val if i == j else (0 if i > j else 1), \
                              -(lgamma(i + alpha / 2) + lgamma(alpha / 2) - lgamma(i + alpha) + (lgamma(j + alpha / 2) + lgamma(alpha / 2) - lgamma(j + alpha)))

            opt_trees.append([])
            for opt_max_depth in opt_depths:
                  clf_opt = DL85Classifier(
                        cache_type=Cache_Type.Cache_HashCover,
                        depth_two_special_algo=False,
                        similar_lb=False,
                        fast_error_lb_function=error_opt,
                        max_depth=opt_max_depth,
                        time_limit=time_limit)
                  start = time.perf_counter()
                  clf_opt.fit(X_train, y_train)
                  duration = time.perf_counter() - start
                  opt_trees[-1].append((clf_opt.tree_, clf_opt.timeout_, duration))

            clf_sparse_opt = DL85Classifier(
                  cache_type=Cache_Type.Cache_HashCover,
                  depth_two_special_algo=False,
                  similar_lb=False,
                  fast_error_lb_function=error_sparse_opt,
                  max_depth=lowest_possible_depth,
                  time_limit=time_limit)
            start = time.perf_counter()
            clf_sparse_opt.fit(X_train, y_train)
            duration = time.perf_counter() - start
            sparse_opt_trees.append((clf_sparse_opt.tree_, clf_sparse_opt.timeout_, duration))
            
            clf_map = DL85Classifier(
                  cache_type=Cache_Type.Cache_HashCover,
                  depth_two_special_algo=False,
                  similar_lb=False,
                  split_penalty_function=split_penalty_map,
                  fast_error_lb_function=error_map,
                  max_depth=lowest_possible_depth,
                  time_limit=time_limit)
            start = time.perf_counter()
            clf_map.fit(X_train, y_train)
            duration = time.perf_counter() - start
            map_trees.append((clf_map.tree_, clf_map.timeout_, duration))

      return opt_trees, sparse_opt_trees, map_trees


if __name__ == "__main__":
      parser = ArgumentParser()
      parser.add_argument('dataset', type=str)
      parser.add_argument('--alpha', type=float, default=5.0)
      parser.add_argument('--alpha_s', type=float, default=0.95)
      parser.add_argument('--beta_s', type=float, default=0.5)
      parser.add_argument('--lamb', type=float, default=0.005)
      parser.add_argument('--opt_depths', type=int, nargs='+', default=[2, 3, 4, 5])
      parser.add_argument('--time_limit', type=int, default=1000)
      parser.add_argument('--k', type=int, default=10)
      parser.add_argument('--seed', type=int, default=42)
      parser.add_argument('--cache_size_threshold', type=float, default=40.0)
      parser.add_argument('--max_features', type=int, default=100)
      args = parser.parse_args()

      opt_trees, sparse_opt_trees, map_trees = run_experiment(
            args.dataset,
            args.alpha,
            args.alpha_s,
            args.beta_s,
            args.lamb,
            args.opt_depths,
            args.time_limit,
            args.k,
            args.seed,
            args.cache_size_threshold,
            args.max_features)
      
      for fold in range(args.k):
            dir_tree = DIR_TREES.format(
                  dataset=args.dataset,
                  k=args.k,
                  time_limit=args.time_limit,
                  seed=args.seed,
                  fold=fold)
            os.makedirs(dir_tree, exist_ok=True)
            for i, opt_max_depth in enumerate(args.opt_depths):
                  opt_tree_path = os.path.join(dir_tree, F_OPT_TREE.format(depth=opt_max_depth))
                  pickle.dump(opt_trees[fold][i], open(opt_tree_path, 'wb'))

            sparse_opt_tree_path = os.path.join(dir_tree, F_SPARSE_OPT_TREE.format(lamb=args.lamb))
            pickle.dump(sparse_opt_trees[fold], open(sparse_opt_tree_path, 'wb'))

            map_tree_path = os.path.join(dir_tree, F_MAP_TREE.format(alpha=args.alpha, alpha_s=args.alpha_s, beta_s=args.beta_s))
            pickle.dump(map_trees[fold], open(map_tree_path, 'wb'))
      


