import numpy as np
import pickle

from sklearn.tree import DecisionTreeClassifier

from main import DIR_TREES, F_DATASET, F_MAP_TREE, F_OPT_TREE, F_SPARSE_OPT_TREE, load_data
from tree import Tree

datasets = [
    "anneal",
    "audiology",
    "australian-credit",
    "breast-wisconsin",
    "diabetes",
    "german-credit",
    "heart-cleveland",
    "hepatitis",
    "hypothyroid",
    "ionosphere",
    "kr-vs-kp",
    "letter",
    "lymph",
    "mushroom",
    "pendigits",
    "primary-tumor",
    "segment",
    "soybean",
    "splice-1",
    "tic-tac-toe",
    "vehicle",
    "vote",
    "yeast",
    "zoo-1"
]


if __name__ == '__main__':

    # bounded by 2^(n/10)
    # bounded by 3^d

    # 

    db_stats = []
    for dataset in datasets:
        data = np.genfromtxt(F_DATASET.format(dataset=dataset), delimiter=' ')
        X, y = data[:, 1:], data[:, 0]
        n = X.shape[0]
        d = X.shape[1]
        db_stats.append((dataset, min((n / 10) * np.log(2), d * np.log(3)) + np.log(n / 10), n, d))
        # db_stats.append((dataset, min(n / 5, d), n, d))
        db_stats = [dbs for dbs in db_stats if dbs[1] <= 40 and dbs[3] <= 100]
    print('\n'.join(map(str, sorted(db_stats, key=lambda x: x[1]))))
    # print(sorted(db_stats).join("\n"))

    exit()


    dataset = "zoo-1"
    data = load_data(dataset)

    K = 10

    total_map_tree_acc = 0.0
    total_opt_tree_acc = 0.0
    total_sparse_opt_tree_acc = 0.0
    total_cart_acc = 0.0
    for tree_fold in range(K):
        print()
        print("Tree fold", tree_fold)
        map_tree_dict, timeout, time = pickle.loads(open(DIR_TREES.format(dataset=dataset, k=K, time_limit=1000, seed=42, fold=tree_fold) + "/" + F_MAP_TREE.format(alpha=2.0, alpha_s=0.95, beta_s=0.5), "rb").read())
        opt_tree_dict, timeout, time = pickle.loads(open(DIR_TREES.format(dataset=dataset, k=K, time_limit=1000, seed=42, fold=tree_fold) + "/" + F_OPT_TREE.format(depth=5), "rb").read())
        sparse_opt_tree_dict, timeout, time = pickle.loads(open(DIR_TREES.format(dataset=dataset, k=K, time_limit=1000, seed=42, fold=tree_fold) + "/" + F_SPARSE_OPT_TREE.format(lamb=0.005), "rb").read())

        print(map_tree_dict)

        map_tree = Tree.from_dict(map_tree_dict)
        opt_tree = Tree.from_dict(opt_tree_dict)
        sparse_opt_tree = Tree.from_dict(sparse_opt_tree_dict)

        X_train = data[tree_fold][0]
        y_train = data[tree_fold][1]

        clf = DecisionTreeClassifier(min_samples_leaf=1, random_state=42)
        clf.fit(X_train, y_train)

        X_test = np.concatenate([data[i][0] for i in range(K) if i != tree_fold])
        y_test = np.concatenate([data[i][1] for i in range(K) if i != tree_fold])

        total_map_tree_acc += map_tree.accuracy(X_test, y_test)
        total_opt_tree_acc += opt_tree.accuracy(X_test, y_test)
        total_sparse_opt_tree_acc += sparse_opt_tree.accuracy(X_test, y_test)
        total_cart_acc += clf.score(X_test, y_test)

    print("Average map tree accuracy:", total_map_tree_acc / K)
    print("Average opt tree accuracy:", total_opt_tree_acc / K)
    print("Average sparse opt tree accuracy:", total_sparse_opt_tree_acc / K)
    print("Average CART accuracy:", total_cart_acc / K)
        