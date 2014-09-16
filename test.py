from __future__ import division

import time

import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot

from scw import SCW1


def generate_dataset():
    digits = load_digits(2)

    classes = np.unique(digits.target)
    y = []
    for target in digits.target:
        if(target == classes[0]):
            y.append(-1)
        if(target == classes[1]):
            y.append(1)
    y = np.array(y)

    return digits.data, y


def calc_accuracy(resutls, answers):
    n_correct_answers = 0
    for result, answer in zip(results, answers):
        if(result == answer):
            n_correct_answers += 1
    accuracy = n_correct_answers/len(results)
    return accuracy

X, y = generate_dataset()

N = int(len(X)*0.8)
training, test = X[:N], X[N:]
labels, answers = y[:N], y[N:]

scw = SCW1(C=1.0, ETA=1.0)
t1 = time.time()
scw.fit(training, labels)
t2 = time.time()
results = scw.predict(test)
accuracy = calc_accuracy(results, answers)
print("SCW    time:{:3.6f}    accuracy:{:1.3f}".format(t2-t1, accuracy))
