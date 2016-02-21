import numpy as np
from sklearn import cross_validation


def overwrite_labels(y):
    classes = np.unique(y)
    y[y==classes[0]] = -1
    y[y==classes[1]] = 1
    return y


def train_test_split(X, y, test_size=0.2):
    data = cross_validation.train_test_split(X, y, test_size=test_size)
    training = data[0], data[2]
    test = data[1], data[3]
    return training, test
