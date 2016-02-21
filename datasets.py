import numpy as np
from sklearn import datasets

import utils


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
