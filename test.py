from __future__ import division

import time

import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot

from scw import SCW1, SCW2


def generate_dataset():
    digits = load_digits(2)

    classes = np.unique(digits.target)
    y = np.array(digits.target, copy=True)
    y[y==classes[0]] = -1
    y[y==classes[1]] = 1
    return digits.data, y


def calc_accuracy(results, answers):
    n_correct_answers = 0
    for result, answer in zip(results, answers):
        if(result == answer):
            n_correct_answers += 1
    accuracy = n_correct_answers/len(results)
    return accuracy


def test_scw(scw, training, test):
    t1 = time.time()
    # online fitting
    for x, y in zip(*training):
        scw.fit([x], [y])
    t2 = time.time()

    samples, labels = test

    t3 = time.time()
    results = scw.predict(samples)
    t4 = time.time()

    accuracy = calc_accuracy(results, labels)

    assert(accuracy == 1.0)

    print("fitting time    : {:3.6f}\n"
          "predicting time : {:3.6f}\n"
          "accuracy        : {:1.3f}\n".format(t2-t1, t4-t3, accuracy))



X, y = generate_dataset()
N = int(len(X)*0.8)

training = (X[:N], y[:N])
test = (X[N:], y[N:])

print("SCW1")
test_scw(SCW1(C=1.0, ETA=1.0), training, test)

print("")
print("SCW2")
test_scw(SCW2(C=1.0, ETA=1.0), training, test)
