import numpy as np
import pickle
import os

from sklearn.tree import DecisionTreeClassifier

from experiments.main import DIR_TREES, F_DATASET, F_MAP_TREE, F_OPT_TREE, F_SPARSE_OPT_TREE, load_data
from tree import Tree
from argparse import ArgumentParser

from typing import Set, FrozenSet, List

from prettytable import PrettyTable

datasets = [
    'australian-un-reduced_converted',
    'bank_conv_categorical_bin',
    'banknote_categorical_bin',
    'cancer-un-reduced_converted',
    'car-un-reduced_converted',
    'cleve-un-reduced_converted',
    'colic-un-reduced_converted',
    'compas-binary',
    'fico-binary',
    'haberman-un-reduced_converted',
    'heart-statlog-un-reduced_converted',
    'hungarian-un-reduced_converted',
    'messidor_categorical_bin',
    'primary-tumor',
    'seismic_bumps_categorical_bin',
    'shuttleM-un-reduced_converted',
    # 'soybean',
    'spect-un-reduced_converted',
    # 'tic-tac-toe',
    # 'vote',
]

SAMPLE_SIZES = [10, 20, 40, 80, 160]
NUM_SAMPLES_TAKEN = 10
OPT_DEPTHS = [2, 3, 4, 5]
SEED = 42

ALPHA = 5.0
ALPHA_S = 0.95
BETA_S = 0.5
LAMB = 0.005

def region_compatibility(F_1: List[FrozenSet[int]], F_2: List[FrozenSet[int]], D: int):
    F_1 = set(F_1)
    F_2 = set(F_2)
    shared_sets = F_1.intersection(F_2)
    F_1 = F_1.difference(shared_sets)
    F_2 = F_2.difference(shared_sets)
    all_sets = list(F_1) + list(F_2)
    if len(all_sets) == 0:
        assert sum(len(st) for st in shared_sets) == D
        return 0.0

    W = np.zeros((len(all_sets), len(all_sets)))
    for i, r_1 in enumerate(all_sets):
        for j, r_2 in enumerate(all_sets):
            W[i, j] = len(r_1.intersection(r_2)) / len(r_1.union(r_2))
    V = np.zeros(len(all_sets))
    for i, r in enumerate(all_sets):
        V[i] = len(r) / D if r in F_1 else -len(r) / D
    return V @ W @ V

def results(dataset: str):
    data, n, d, l = load_data(dataset, SAMPLE_SIZES, NUM_SAMPLES_TAKEN, SEED)
    print(dataset)

    values = np.zeros((len(SAMPLE_SIZES), 7, 4))
    for i, sample_size in enumerate(SAMPLE_SIZES):
        print(f"Sample Size: {sample_size}")
        opt_trees = []
        sparse_opt_trees = []
        map_trees = []
        cart_trees = []
        for j in range(NUM_SAMPLES_TAKEN):
            dir_tree = DIR_TREES.format(
                dataset=dataset,
                seed=SEED,
                idx=j,
                sample_size=sample_size)

            opt_trees.append([])
            for d, opt_max_depth in enumerate(OPT_DEPTHS):
                opt_tree_file = F_OPT_TREE.format(depth=opt_max_depth)
                opt_tree_path = os.path.join(dir_tree, opt_tree_file)
                opt_tree_dict, duration = pickle.loads(open(opt_tree_path, 'rb').read())
                opt_trees[j].append(Tree.from_dict(opt_tree_dict))

            sparse_opt_tree_file = F_SPARSE_OPT_TREE.format(lamb=LAMB)
            sparse_opt_tree_path = os.path.join(dir_tree, sparse_opt_tree_file)
            sparse_opt_tree_dict, duration = pickle.loads(open(sparse_opt_tree_path, 'rb').read())
            sparse_opt_trees.append(Tree.from_dict(sparse_opt_tree_dict))

            map_tree_file = F_MAP_TREE.format(alpha=ALPHA, alpha_s=ALPHA_S, beta_s=BETA_S)
            map_tree_path = os.path.join(dir_tree, map_tree_file)
            map_tree_dict, duration = pickle.loads(open(map_tree_path, 'rb').read())
            map_trees.append(Tree.from_dict(map_tree_dict))

            cart_tree = DecisionTreeClassifier(random_state=SEED)
            X_train, _, y_train, _ = data[i][j]
            cart_tree.fit(X_train, y_train)
            cart_trees.append(Tree.from_sklearn_tree(cart_tree.tree_))

        # accuracy
        total_opt_tree_accs = [0.0] * len(OPT_DEPTHS)
        total_sparse_opt_tree_acc = 0.0
        total_map_tree_acc = 0.0
        total_cart_tree_acc = 0.0

        # log likelihood
        total_opt_tree_lls = [0.0] * len(OPT_DEPTHS)
        total_sparse_opt_tree_ll = 0.0
        total_map_tree_ll = 0.0
        total_cart_tree_ll = 0.0

        # regional compatibility
        total_opt_tree_stabs = [0.0] * len(OPT_DEPTHS)
        total_sparse_opt_tree_stab = 0.0
        total_map_tree_stab = 0.0
        total_cart_tree_stab = 0.0

        # semantic similarity
        total_opt_tree_sims = [0.0] * len(OPT_DEPTHS)
        total_sparse_opt_tree_sim = 0.0
        total_map_tree_sim = 0.0
        total_cart_tree_sim = 0.0

        for j in range(NUM_SAMPLES_TAKEN):
            X_train, X_test, y_train, y_test = data[i][j]
            X = np.concatenate((X_train, X_test), axis=0)

            for d, opt_max_depth in enumerate(OPT_DEPTHS):
                opt_tree = opt_trees[j][d]
                total_opt_tree_accs[d] += opt_tree.accuracy(X_test, y_test)
                total_opt_tree_lls[d] += opt_tree.log_likelihood(X_test, y_test, alpha=ALPHA)

                for l in range(j + 1, NUM_SAMPLES_TAKEN):
                    if l == j:
                        continue
                    F_1 = opt_trees[j][d].regions(X)
                    F_2 = opt_trees[l][d].regions(X)
                    total_opt_tree_stabs[d] += region_compatibility(F_1, F_2, X.shape[0])
                    total_opt_tree_sims[d] += np.mean(opt_trees[j][d].predict(X) == opt_trees[l][d].predict(X))

            sparse_opt_tree = sparse_opt_trees[j]
            total_sparse_opt_tree_acc += sparse_opt_tree.accuracy(X_test, y_test)
            total_sparse_opt_tree_ll += sparse_opt_tree.log_likelihood(X_test, y_test, alpha=ALPHA)
            for l in range(j + 1, NUM_SAMPLES_TAKEN):
                if l == j:
                    continue
                F_1 = sparse_opt_trees[j].regions(X)
                F_2 = sparse_opt_trees[l].regions(X)
                total_sparse_opt_tree_stab += region_compatibility(F_1, F_2, X.shape[0])
                total_sparse_opt_tree_sim += np.mean(sparse_opt_trees[j].predict(X) == sparse_opt_trees[l].predict(X))

            map_tree = map_trees[j]
            total_map_tree_acc += map_tree.accuracy(X_test, y_test)
            total_map_tree_ll += map_tree.log_likelihood(X_test, y_test, alpha=ALPHA)
            for l in range(j + 1, NUM_SAMPLES_TAKEN):
                if l == j:
                    continue
                F_1 = map_trees[j].regions(X)
                F_2 = map_trees[l].regions(X)
                total_map_tree_stab += region_compatibility(F_1, F_2, X.shape[0])
                total_map_tree_sim += np.mean(map_trees[j].predict(X) == map_trees[l].predict(X))

            cart_tree = cart_trees[j]
            total_cart_tree_acc += cart_tree.accuracy(X_test, y_test)
            total_cart_tree_ll += cart_tree.log_likelihood(X_test, y_test, alpha=ALPHA)
            for l in range(j + 1, NUM_SAMPLES_TAKEN):
                if l == j:
                    continue
                F_1 = cart_trees[j].regions(X)
                F_2 = cart_trees[l].regions(X)
                total_cart_tree_stab += region_compatibility(F_1, F_2, X.shape[0])
                total_cart_tree_sim += np.mean(cart_trees[j].predict(X) == cart_trees[l].predict(X))
            
        avg_map_tree_acc = total_map_tree_acc / NUM_SAMPLES_TAKEN
        avg_map_tree_ll = total_map_tree_ll / NUM_SAMPLES_TAKEN
        avg_map_tree_stab = total_map_tree_stab / (NUM_SAMPLES_TAKEN * (NUM_SAMPLES_TAKEN - 1) / 2)
        avg_map_tree_sim = total_map_tree_sim / (NUM_SAMPLES_TAKEN * (NUM_SAMPLES_TAKEN - 1) / 2)

        avg_sparse_opt_tree_acc = total_sparse_opt_tree_acc / NUM_SAMPLES_TAKEN
        avg_sparse_opt_tree_ll = total_sparse_opt_tree_ll / NUM_SAMPLES_TAKEN
        avg_sparse_opt_tree_stab = total_sparse_opt_tree_stab / (NUM_SAMPLES_TAKEN * (NUM_SAMPLES_TAKEN - 1) / 2)
        avg_sparse_opt_tree_sim = total_sparse_opt_tree_sim / (NUM_SAMPLES_TAKEN * (NUM_SAMPLES_TAKEN - 1) / 2)

        avg_cart_tree_acc = total_cart_tree_acc / NUM_SAMPLES_TAKEN
        avg_cart_tree_ll = total_cart_tree_ll / NUM_SAMPLES_TAKEN
        avg_cart_tree_stab = total_cart_tree_stab / (NUM_SAMPLES_TAKEN * (NUM_SAMPLES_TAKEN - 1) / 2)
        avg_cart_tree_sim = total_cart_tree_sim / (NUM_SAMPLES_TAKEN * (NUM_SAMPLES_TAKEN - 1) / 2)

        avg_opt_tree_accs = [0.0] * len(OPT_DEPTHS)
        avg_opt_tree_lls = [0.0] * len(OPT_DEPTHS)
        avg_opt_tree_stabs = [0.0] * len(OPT_DEPTHS)
        avg_opt_tree_sims = [0.0] * len(OPT_DEPTHS)
        for d, opt_max_depth in enumerate(OPT_DEPTHS):
            avg_opt_tree_accs[d] = total_opt_tree_accs[d] / NUM_SAMPLES_TAKEN
            avg_opt_tree_lls[d] = total_opt_tree_lls[d] / NUM_SAMPLES_TAKEN
            avg_opt_tree_stabs[d] = total_opt_tree_stabs[d] / (NUM_SAMPLES_TAKEN * (NUM_SAMPLES_TAKEN - 1) / 2)
            avg_opt_tree_sims[d] = total_opt_tree_sims[d] / (NUM_SAMPLES_TAKEN * (NUM_SAMPLES_TAKEN - 1) / 2)

        values[i][0][0] = avg_map_tree_acc
        values[i][0][1] = avg_map_tree_ll
        values[i][0][2] = avg_map_tree_stab
        values[i][0][3] = avg_map_tree_sim

        values[i][1][0] = avg_sparse_opt_tree_acc
        values[i][1][1] = avg_sparse_opt_tree_ll
        values[i][1][2] = avg_sparse_opt_tree_stab
        values[i][1][3] = avg_sparse_opt_tree_sim

        values[i][2][0] = avg_cart_tree_acc
        values[i][2][1] = avg_cart_tree_ll
        values[i][2][2] = avg_cart_tree_stab
        values[i][2][3] = avg_cart_tree_sim

        for d, opt_max_depth in enumerate(OPT_DEPTHS):
            values[i][3 + d][0] = avg_opt_tree_accs[d]
            values[i][3 + d][1] = avg_opt_tree_lls[d]
            values[i][3 + d][2] = avg_opt_tree_stabs[d]
            values[i][3 + d][3] = avg_opt_tree_sims[d]
    
    return values


if __name__ == '__main__':
    
    # values = np.zeros((len(SAMPLE_SIZES), 7, 4))
    # if not os.path.exists("results"):
    #     os.makedirs("results")
    
    # for dataset in datasets:
    #     values = results(dataset)
    #     np.save(f"results/{dataset}.npy", values)

    wins = np.zeros((len(SAMPLE_SIZES), 6, 4))
    diff = np.zeros((len(SAMPLE_SIZES), 6, 4))
    for dataset in datasets:
        values = np.load(f"results/{dataset}.npy")
        wins += values[:, 0, :].reshape(len(SAMPLE_SIZES), 1, 4) > values[:, 1:, :]
        diff += values[:, 0, :].reshape(len(SAMPLE_SIZES), 1, 4) - values[:, 1:, :]

    print("Win %")
    for i, sample_size in enumerate(SAMPLE_SIZES):
        print(f"Sample size: {sample_size}")
        table = PrettyTable(field_names=["Metric", "MAP vs. SparseOpt", "MAP vs. CART", "MAP vs. Opt_2", "MAP vs. Opt_3", "MAP vs. Opt_4", "MAP vs. Opt_5"])
        for j, metric in enumerate(["Accuracy", "Log Likelihood", "Stability", "Similarity"]):
            table.add_row([metric] + [f"{prc}%" for prc in np.round(wins[i, :, j] / len(datasets) * 100, 2)])
        print(table)
