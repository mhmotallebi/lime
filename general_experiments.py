"""
A helper script for evaluating performance of changes to the tabular explainer, in this case different
implementations and methods for distance calculation.
"""

import argparse
import random
import time
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

from lime.lime_tabular import LimeTabularExplainer

RANDOM_SEED=1
random.seed(RANDOM_SEED)

def get_tree_explanation(tree, v):
    """From LIME repo, with minor modificaitons"""
    t = tree.tree_
    nonzero = v.nonzero()[0]
    current = 0
    left_child = t.children_left[current]
    exp = set()
    while left_child != sklearn.tree._tree.TREE_LEAF:
        left_child = t.children_left[current]
        right_child = t.children_right[current]
        f = t.feature[current]
        if f in nonzero:
            exp.add(f)
        if v[f] < t.threshold[current]:
            current = left_child
        else:
            current = right_child
    return exp

def interpret_data_old(X, y, func):
    explainer = LimeTabularExplainer(X, discretize_continuous=False, kernel_width=10)
    times, scores = [], []
    for r_idx in range(100):
        start_time = time.time()
        explanation = explainer.explain_instance(X[r_idx, :], func)
        times.append(time.time() - start_time)
        scores.append(explanation.score)

    return times, scores

def get_decision_path(clf, instance):
    """Works same as the LIME function"""
    feature = clf.tree_.feature
    leave_id = clf.apply(instance.reshape(1, -1))
    node_indicator = clf.decision_path([instance])
    features = []
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    for node_id in node_index:
        if leave_id[0] == node_id:  # <-- changed != to ==
            continue # <-- comment out
        else: # < -- added else to iterate through decision nodes
            features.append(feature[node_id])
    features = set(features)
    if verbose:
        print('tree features:', features)
    return features


def get_lime_features(explanation, length):
    tuples = explanation.as_map()[explanation.available_labels()[0]]
    if verbose:
        print(tuples)
    features = [x[0] for x in sorted(tuples, key=lambda x:x[1], reverse=True)][:length]
    if verbose:
        print('lime features:', features)
    return features

def interpret_data(X, y, func, clf, samples_per_instance, n_features_lime, make_discretize, discretizer):
    # print('clf.classes:', clf.classes_)
    # labels = np.argmax(y, axis=1)
    labels = np.vectorize(np.argmax)(y)
    explainer = LimeTabularExplainer(X, training_labels=labels, discretize_continuous=make_discretize, discretizer=discretizer)
    # print(explainer.__dict__)
    times, scores = [], []
    for r_idx in range(100):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        if verbose:
            print()
        
        start_time = time.time()
        explanation = explainer.explain_instance(X[r_idx, :], func, 
                    num_features=n_features_lime, num_samples=samples_per_instance)
        times.append(time.time() - start_time)

        decision_path = get_tree_explanation(clf, X[r_idx, :])
        # decision_path = get_decision_path(clf, X[r_idx, :])
        
        if verbose:
            print('probs on train set:',y[r_idx])
        lime_features = get_lime_features(explanation, n_features_lime)
        score = len(set(decision_path).intersection(lime_features))/len(decision_path)
        if verbose:
            print('score:',score)
        scores.append(score)

    return times, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, help="how many classes should the dataset have", default=2)
    parser.add_argument("--n_features_dataset", type=int, help="how many features should the dataset have", default=100)
    parser.add_argument("--n_samples_dataset", type=int, help="how many samples should training set of decision tree have", default=10000)

    parser.add_argument("--n_samples_per_instance", type=int, help="how many samples should LIME make to explain an instnace", default=15000)
    parser.add_argument("--discretizer", choices=["entropy", "decile", "quartile"], default=None)

    parser.add_argument("--dt_max_depth", type=int, help='what should the maximum depth of the decision tree be', default=10)
    parser.add_argument("--n_features_lime", type=int, help="how many features should a lime explanation have", default=10)

    parser.add_argument("-v", "--verbose", action="store_true", help="add verbosity", default=False)
    args = parser.parse_args()
    print(args)

    verbose = args.verbose
    dt_max_depth = args.dt_max_depth
    n_classes = args.n_classes
    n_features_dataset = args.n_features_dataset
    n_samples_dataset = args.n_samples_dataset
    n_samples_per_instance = args.n_samples_per_instance
    n_features_lime = args.n_features_lime
    discretizer = args.discretizer
    make_discretize = True if discretizer is not None else False
    X_raw, y_raw = make_classification(n_classes=n_classes, n_features=n_features_dataset, 
                                        n_samples=n_samples_dataset, random_state=RANDOM_SEED)
    clf = DecisionTreeClassifier(max_depth=dt_max_depth, random_state=RANDOM_SEED)
    clf.fit(X_raw, y_raw)

    y_hat = clf.predict(X_raw)
    print('clf accuracy on train set:', sum(x==y for (x,y) in zip(y_hat, y_raw))/len(y_hat))

    y_hat = clf.predict_proba(X_raw)
    # print(np.argmax(y_hat))
    times, scores = interpret_data(X_raw, y_hat, clf.predict_proba, clf, n_samples_per_instance, 
                                                    n_features_lime, make_discretize, discretizer)

    print('%9.4fs %9.4fs %9.4fs' % (min(times), sum(times) / len(times), max(times)))
    print('%9.4f %9.4f% 9.4f' % (min(scores), sum(scores) / len(scores), max(scores)))