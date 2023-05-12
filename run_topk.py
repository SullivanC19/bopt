from multiprocessing import Process
import pickle

import numpy as np
import pandas as pd

from argparse import ArgumentParser

from pydl85 import DL85Classifier, Cache_Type

from topk.dataloader import load_data, load_data_numerical, load_data_numerical_tt_split

CATEGORICAL_DATA_PATH = "topk/data/categorical"
NUMERICAL_DATA_PATH = "topk/data/numerical"
TEST_TRAIN_DATA_PATH = "topk/data/numerical"

CATEGORICAL_DATASETS = [
    "audiology",
    "balance-scale",
    "breast-cancer",
    "car",
    "connect-4",
    "hayes-roth",
    "hiv-1-protease",
    "kr-vs-kp",
    "led-display",
    "letter-recognition",
    "lymph",
    "molecular-biology-promoters",
    "molecular-biology-splice",
    "monks-1",
    "monks-2",
    "monks-3",
    "nursery",
    "primary-tumor",
    "rice",
    "soybean",
    "spect",
    "tic-tac-toe",
    "vote",
    "wireless-indoor",
    "yeast",
]

NUMERICAL_DATASETS = [
    "artificial-characters",
    "credit-card",
    "dry-bean",
    "electrical-grid-stability",
    "miniboone",
    "occupancy-estimation",
    "sensorless-drive-diagnosis",
    "spambase",
    "taiwanese-bankruptcy",
    "telescope",
]

TEST_TRAIN_DATASETS = [
    "avila",
    "ml-prove",
]

def run_topk(data, k, maxdepth):
    X_train, y_train, X_test, y_test, _, _ = data
    X_train = np.array(X_train).astype(np.int32)
    X_test = np.array(X_test).astype(np.int32)
    y_train = np.array(y_train).astype(np.int32)
    y_test = np.array(y_test).astype(np.int32)

    clf = DL85Classifier(
        k=k,
        max_depth=maxdepth,
        depth_two_special_algo=False,
        similar_lb=False,
        similar_for_branching=False,
        desc=True,
        repeat_sort=True,
        cache_type=Cache_Type.Cache_HashCover,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    time_taken = clf.runtime_
    tree = clf.base_tree_

    return tree, train_acc, test_acc, time_taken

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('maxdepth', type=int)
    parser.add_argument('--k', type=int, default=0) # 0 means k = n
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()

    data = None
    if args.dataset in CATEGORICAL_DATASETS:
        data = load_data(CATEGORICAL_DATA_PATH, args.dataset, seed=args.seed)
    elif args.dataset in NUMERICAL_DATASETS:
        data = load_data_numerical(NUMERICAL_DATA_PATH, args.dataset, seed=args.seed)
    elif args.dataset in TEST_TRAIN_DATASETS:
        data = load_data_numerical_tt_split(TEST_TRAIN_DATA_PATH, args.dataset, max_splits=100)
    else:
        raise ValueError("Invalid dataset")

    tree, train_acc, test_acc, time_taken = run_topk(data, args.k, args.maxdepth)
    
    print(f"Train acc: {train_acc}")
    print(f"Test acc: {test_acc}")
    print(f"Time taken: {time_taken}")

    print("Tree:")
    print(tree)
