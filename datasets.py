from os.path import exists, join

import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle

import utils


dataset_dir = "datasets"


def fetch_mnist(training_ratio=0.8, data_home="."):
    mnist = datasets.fetch_mldata('MNIST original', data_home=data_home)

    X, y = shuffle(mnist.data, mnist.target)

    splitter = int(len(X)*training_ratio)
    training = X[:splitter], y[:splitter]
    test = X[splitter:], y[splitter:]
    return training, test


def download_mnist():
    training, test = fetch_mnist(data_home=dataset_dir)

    X, y = training
    datasets.dump_svmlight_file(X, y, join(dataset_dir, "mnist"))

    X, y = test
    datasets.dump_svmlight_file(X, y, join(dataset_dir, "mnist.t"))


def load_mnist():
    def pick(X, y):
        indices = np.logical_or(y==0, y==1)
        X = X.todense()
        X = X[indices]
        y = y[indices]
        y = utils.overwrite_labels(y)
        return X, y

    n_features = 784
    training_path = join(dataset_dir, "mnist")
    test_path = join(dataset_dir, "mnist.t")

    if not exists(training_path) or not exists(test_path):
        download_mnist()

    X, y = datasets.load_svmlight_file(training_path, n_features=n_features)
    training = pick(X, y)

    X, y = datasets.load_svmlight_file(test_path, n_features=n_features)
    test = pick(X, y)
    return training, test


def make_classification():
    X, y = datasets.make_classification(n_samples=5000, n_features=100,
                                        n_classes=2)
    y = utils.overwrite_labels(y)  # (0, 1) to (-1, 1)
    return X, y


def load_digits():
    digits = datasets.load_digits(2)
    y = np.array(digits.target, copy=True)
    y = utils.overwrite_labels(y)
    return digits.data, y
