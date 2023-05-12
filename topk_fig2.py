import os
from multiprocessing import Process
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from argparse import ArgumentParser

from tqdm import tqdm

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
    
SEEDS = list(range(42, 52))

k_values = [1,2,3,4,8,12,16,0]
max_depths = [3,4,5,6,7,8]

TRAIN_ACC_PATH = "topk/results/accuracy/dataset={dataset}_seed={seed}-train.npy"
TEST_ACC_PATH = "topk/results/accuracy/dataset={dataset}_seed={seed}-test.npy"

TREE_AND_RESULTS_PATH = "topk/results/trees/dataset={dataset}_k={k}_maxdepth={maxdepth}_seed={seed}.pkl"

CATEGORICAL_RESULTS_PATH = "topk/results/categorical"
NUMERICAL_RESULTS_PATH = "topk/results/numerical"
TOP_K_RESULTS_FILE = "top{k}.csv"

def run_topk_experiment(dataset, data, k, maxdepth, seed):
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
    pickle.dump((clf.tree_, train_acc, test_acc), open(TREE_AND_RESULTS_PATH.format(dataset=dataset, k=k, maxdepth=maxdepth, seed=seed), "wb"))

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("idx", type=int)
    # parser.add_argument("--k", type=int, default=1)
    # parser.add_argument("--maxdepth", type=int, default=3)
    # parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()

    idx = args.idx

    seed = SEEDS[idx % len(SEEDS)]
    idx //= len(SEEDS)

    maxdepth = max_depths[idx % len(max_depths)]
    idx //= len(max_depths)

    k = k_values[idx % len(k_values)]
    idx //= len(k_values)

    data = None
    dataset = None
    if idx < len(CATEGORICAL_DATASETS):
        dataset = CATEGORICAL_DATASETS[idx]
        data = load_data(CATEGORICAL_DATA_PATH, dataset, seed=seed)
    else:
        idx -= len(CATEGORICAL_DATASETS)
    
        if idx < len(NUMERICAL_DATASETS):
            dataset = NUMERICAL_DATASETS[idx]
            data = load_data_numerical(NUMERICAL_DATA_PATH, dataset, seed=seed)
        else:
            idx -= len(NUMERICAL_DATASETS)

            if idx < len(TEST_TRAIN_DATASETS):
                dataset = TEST_TRAIN_DATASETS[idx]
                data = load_data_numerical_tt_split(TEST_TRAIN_DATA_PATH, dataset, max_splits=100)
                if seed != SEEDS[0]:
                    exit() # train/test split won't change with seed
            else:
                raise ValueError("Invalid idx")
            
    print("Dataset: {}".format(dataset))
    print("Seed: {}".format(seed))
    print("Max depth: {}".format(maxdepth))
    print("k: {}".format(k))

    run_topk_experiment(dataset, data, k, maxdepth, seed)
