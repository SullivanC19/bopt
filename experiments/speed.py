import numpy as np
from math import lgamma, log

from argparse import ArgumentParser

from pydl85 import DL85Classifier, Cache_Type
from tree import Tree

DATASETS = [
    'primary-tumor',
    'australian-un-reduced_converted'
]

F_OUT = 'map_speed.out'

ALPHA = 5.0
ALPHA_S = 0.95
BETA_S = 0.5

def run(dataset, max_depth):
    data = np.genfromtxt(f'datasets/{dataset}.txt')
    X, y = data[:, 1:], data[:, 0]

    lowest_possible_depth = len(X)
    tiebreak_val = np.argmax(np.bincount(y.astype(np.int32)))
    all_depths = np.arange(lowest_possible_depth + 1)
    # split_pen = np.log(ALPHA_S) - BETA_S * np.log(1 + all_depths)
    split_pen = np.log(ALPHA_S) - BETA_S * np.log(1 + all_depths)
    stop_pen = np.log(1 - ALPHA_S * np.power(1 + all_depths, -BETA_S))
    alpha_prior_term = lgamma(ALPHA) - 2 * lgamma(ALPHA / 2)

    def split_penalty_map(depth, splits):
        return 0 if splits == 0 else -(-stop_pen[depth] + split_pen[depth] + 2 * stop_pen[depth + 1] - log(splits))

    def error_map(label_counts):
        c0, c1 = label_counts
        leaf_error = -(lgamma(c0 + ALPHA / 2) + lgamma(c1 + ALPHA / 2) - lgamma(c0 + c1 + ALPHA) + alpha_prior_term)
        perfect_split_error = -((lgamma(c0 + ALPHA / 2) + lgamma(ALPHA / 2) - lgamma(c0 + ALPHA)) + (lgamma(c1 + ALPHA / 2) + lgamma(ALPHA / 2) - lgamma(c1 + ALPHA)) + 2 * alpha_prior_term)
        return leaf_error, tiebreak_val if c0 == c1 else (0 if c0 > c1 else 1), min(leaf_error, perfect_split_error)

    clf = DL85Classifier(
        cache_type=Cache_Type.Cache_HashCover,
        depth_two_special_algo=False,
        similar_lb=False,
        similar_for_branching=False,
        split_penalty_function=split_penalty_map,
        fast_error_lb_function=error_map,
        max_depth=max_depth if max_depth != -1 else lowest_possible_depth,
        desc=True,
        repeat_sort=True)
    clf.fit(X, y)

    return -(clf.error_ - stop_pen[0]), clf.runtime_


if __name__ == '__main__':
    with open(F_OUT, 'a') as f:
        for dataset in DATASETS:
            for depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1]:
                error, runtime = run(dataset, depth)
                f.write(f'{dataset} {depth} {error} {runtime}\n')
