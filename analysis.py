import numpy as np
import pickle

from sklearn.tree import DecisionTreeClassifier

from main import DIR_TREES, F_DATASET, F_MAP_TREE, F_OPT_TREE, F_SPARSE_OPT_TREE, load_data
from tree import Tree
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    data, n, d, l = load_data(args.dataset)

    print(args.dataset)

    K = 10
    DEPTHS = [2, 3, 4, 5]
    ALPHA = 5.0
    ALPHA_S = 0.95
    BETA_S = 0.5
    LAMB = 0.005
    TIME_LIMIT = 1000
    SEED = 42

    total_map_tree_acc = 0.0
    total_opt_tree_accs = [0.0 for _ in range(len(DEPTHS))]
    total_sparse_opt_tree_acc = 0.0
    total_cart_acc = 0.0
    for tree_fold in range(K):
        map_tree_dict, timeout, time = pickle.loads(open(DIR_TREES.format(dataset=args.dataset, k=K, time_limit=TIME_LIMIT, seed=SEED, fold=tree_fold) + "/" + F_MAP_TREE.format(alpha=ALPHA, alpha_s=ALPHA_S, beta_s=BETA_S), "rb").read())
        opts = []
        for depth in DEPTHS:
            opts.append(pickle.loads(open(DIR_TREES.format(dataset=args.dataset, k=K, time_limit=TIME_LIMIT, seed=SEED, fold=tree_fold) + "/" + F_OPT_TREE.format(depth=depth), "rb").read()))
        sparse_opt_tree_dict, timeout, time = pickle.loads(open(DIR_TREES.format(dataset=args.dataset, k=K, time_limit=TIME_LIMIT, seed=SEED, fold=tree_fold) + "/" + F_SPARSE_OPT_TREE.format(lamb=LAMB), "rb").read())

        map_tree = Tree.from_dict(map_tree_dict)
        opt_trees = []
        for i in range(len(opts)):
            opt_trees.append(Tree.from_dict(opts[i][0]))
        sparse_opt_tree = Tree.from_dict(sparse_opt_tree_dict)

        X_train = data[tree_fold][0]
        y_train = data[tree_fold][1]

        clf = DecisionTreeClassifier(min_samples_leaf=1, random_state=42)
        clf.fit(X_train, y_train)

        X_test = np.concatenate([data[i][0] for i in range(K) if i != tree_fold])
        y_test = np.concatenate([data[i][1] for i in range(K) if i != tree_fold])

        total_map_tree_acc += map_tree.accuracy(X_test, y_test)
        for i in range(len(opts)):
            total_opt_tree_accs[i] += opt_trees[i].accuracy(X_test, y_test)
        total_sparse_opt_tree_acc += sparse_opt_tree.accuracy(X_test, y_test)
        total_cart_acc += clf.score(X_test, y_test)

    print("Average map tree accuracy:", total_map_tree_acc / K)
    for i in range(len(DEPTHS)):
        print("Average opt tree accuracy (depth {}):".format(DEPTHS[i]), total_opt_tree_accs[i] / K)
    print("Average sparse opt tree accuracy:", total_sparse_opt_tree_acc / K)
    print("Average CART accuracy:", total_cart_acc / K)
        