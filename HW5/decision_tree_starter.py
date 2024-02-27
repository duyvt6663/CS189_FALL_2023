from collections import Counter

import io
import numpy as np
from numpy import genfromtxt
import pandas as pd
from pydot import graph_from_dot_data
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
import sys
import os

eps = 1e-5  # a small number


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        # TODO implement entropy function
        # count the number of each class
        counts = Counter(y)

        # calculate the entropy
        entropy = 0
        for count in counts.values():
            p = count / len(y)
            entropy -= p * np.log2(p)

        return entropy

    @staticmethod
    def information_gain(X, y, thresh):
        '''
        :param X: feature values, shape (n,)
        :param y: labels, shape (n,)
        :param thresh: threshold to split on
        :return: information gain of the split
        '''
        # TODO implement information gain function
        # split the data    
        idx0 = np.where(X < thresh)[0]
        idx1 = np.where(X >= thresh)[0]
        y0, y1 = y[idx0], y[idx1]

        # calculate the entropy of the split
        entropy_split = (len(y0) * DecisionTree.entropy(y0) + len(y1) * DecisionTree.entropy(y1)) / len(y)

        # calculate the information gain
        information_gain = DecisionTree.entropy(y) - entropy_split

        return information_gain

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(
                    np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([
                    self.information_gain(X[:, i], y, t) for t in thresh[i, :]
                ])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(
                np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(
                X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y, keepdims=True).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y, keepdims=True).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(
                X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO implement function
        for tree in self.decision_trees:
            # first, sample data points with replacement from X till 
            # we have the same number of data points as X
            idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            # then fit the tree to the sampled data
            tree.fit(X[idx, :], y[idx])

    def predict(self, X):
        # TODO implement function
        predictions = np.zeros((X.shape[0], self.n))
        for i, tree in enumerate(self.decision_trees):
            predictions[:, i] = tree.predict(X)

        # return the mode of the predictions
        return stats.mode(predictions, axis=1, keepdims=True).mode.ravel()

class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        # TODO implement function
        self.params = params
        self.n = n
        self.m = m
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params, max_features=self.m)
            for i in range(self.n)
        ]


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i], keepdims=True).mode[0]
            data[(data[:, i] > -1 - eps) *
                 (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf, num_splits=3):
    print("Cross validation", cross_val_score(clf, X, y, cv=num_splits))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


def generate_submission(testing_data, predictions):
    # This code below will generate the predictions.csv file.
    if isinstance(predictions, np.ndarray):
        predictions = predictions.astype(int)
    else:
        predictions = np.array(predictions, dtype=int)
    assert predictions.shape == (len(testing_data),), "Predictions were not the correct shape"
    df = pd.DataFrame({'Category': predictions})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv('predictions.csv', index_label='Id')

    # Now download the predictions.csv file to submit.`

def test_information_gain():
    # TODO test information gain function
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    thresh = 5
    information_gain = DecisionTree.information_gain(X, y, thresh)
    print(f"Information gain: {information_gain}")

    expected_total_entropy = -np.log2(0.5) 
    expected_entropy_split = 3/5 * (-5/6 * np.log2(5/6) - 1/6 * np.log2(1/6)) 
    expected_information_gain = expected_total_entropy - expected_entropy_split
    print(f"Expected information gain: {expected_information_gain}")
    assert np.abs(information_gain - expected_information_gain) < 1e-5

if __name__ == "__main__":
    dataset = "spam"
    params = {
        "max_depth": 3,
        "min_samples_leaf": 10,
    }
    N = 200

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # Basic decision tree
    print("\n\nPart (a-b): simplified decision tree")
    # TODO
    dt = DecisionTree(max_depth=5, feature_labels=features) 
    dt.fit(X, y)
    predictions = dt.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy}")

    # Basic decision tree
    print("\n\nPart (c): sklearn's decision tree")
    # Hint: Take a look at the imports!
    clf = DecisionTreeClassifier(random_state=0, **params)
    # TODO
    clf.fit(X, y)
    print(clf.score(X, y))
    # Visualizing the tree
    out = io.StringIO()
    export_graphviz(
        clf, out_file=out, feature_names=features, class_names=class_names)
    # For OSX, may need the following for dot: brew install gprof2dot
    graph = graph_from_dot_data(out.getvalue())
    graph_from_dot_data(out.getvalue())[0].write_pdf("%s-basic-tree.pdf" % dataset)

    # Bagged trees
    print("\n\nPart (d-e): bagged trees")
    # TODO
    clf = BaggedTrees(params=params, n=N)
    clf.fit(X, y)
    predictions = clf.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy}")
    # Gather the first internal node split for each tree
    first_splits = []
    for tree in clf.decision_trees:
        feature_index = tree.tree_.feature[0]
        feature_name = features[feature_index]
        threshold = tree.tree_.threshold[0]
        first_splits.append((feature_name, threshold))
    counter = Counter(first_splits)
    print(f"Most common root split: {counter.most_common(5)}")

    # Random forest
    print("\n\nPart (f-g): random forest")
    # TODO
    clf = RandomForest(params=params, n=N, m=1)
    clf.fit(X, y)
    predictions = clf.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy}")

    # Generate csv file of predictions on test data
    m = 4 if dataset == "titanic" else 20
    models = [
        (DecisionTreeClassifier(random_state=0, **params), "basic-tree"),
        (BaggedTrees(params=params, n=N), "bagged-trees"),
        (RandomForest(params=params, n=N, m=m), "random-forest")
    ]
    save_file = "models_stats.csv"
    df = pd.read_csv(save_file) if os.path.exists(save_file) else pd.DataFrame()
    if "dataset" not in df.columns:
        df['dataset'] = [''] * len(models) * 2
    if "model" not in df.columns:
        df['model'] = [np.nan] * len(models) * 2
    if "train_acc" not in df.columns:
        df['train_acc'] = [np.nan] * len(models) * 2

    for i, (clf, name) in enumerate(models):
        if dataset == "titanic":
            i += 3
        else:
            i += 0
        print(f"Model {name}")
        cross_scores = cross_val_score(clf, X, y, cv=3)
        print(f"Cross validation accuracy: {cross_scores}")

        # Store the validation accuracy in a file
        df['model'][i] = name
        df['dataset'][i] = dataset
        for j, score in enumerate(cross_scores):
            if f'cv_{j}_acc' not in df.columns:
                df[f'cv_{j}_acc'] = [np.nan] * len(models) * 2
            df[f'cv_{j}_acc'][i] = score
        
        clf.fit(X, y)
        df['train_acc'][i] = np.mean(clf.predict(X) == y)
    
    df.to_csv(save_file, index=False)
