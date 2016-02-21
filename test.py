from __future__ import division

import time
import unittest

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot

import utils
from scw import SCW1, SCW2


def generate_dataset():
    digits = load_digits(2)
    y = np.array(digits.target, copy=True)
    y = utils.overwrite_labels(y)
    return digits.data, y


def cross_validation(scw, training, test):
    # force online fitting
    for x, y in zip(*training):
        scw.fit([x], [y])

    X, y_true = test
    y_pred = scw.predict(X)
    return accuracy_score(y_true, y_pred)


class TestAccuracy(unittest.TestCase):
    def test_accuracy(self):
        X, y = generate_dataset()  #linear separable
        N = int(len(X)*0.8)

        training = X[:N], y[:N]
        test = X[N:], y[N:]

        accuracy = cross_validation(SCW1(C=1.0, ETA=1.0), training, test)
        self.assertEqual(accuracy, 1.0)
        accuracy = cross_validation(SCW2(C=1.0, ETA=1.0), training, test)
        self.assertEqual(accuracy, 1.0)


class TestDataFormat(unittest.TestCase):
    def setUp(self):
        self.scw = SCW1()

    def test_data_shape(self):
        X = np.arange(144).reshape(4, 4, 9)  #3 dim array
        y = np.random.randint(0, 1, 144)
        self.assertRaises(ValueError, self.scw.fit, X, y)

    def test_data_label(self):
        X = np.arange(24).reshape(6, 4)
        y = [1, 1, 0, 0, 1, 0]
        self.assertRaises(ValueError, self.scw.fit, X, y)


unittest.main()
