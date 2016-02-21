import numpy as np


def overwrite_labels(y):
    classes = np.unique(y)
    y[y==classes[0]] = -1
    y[y==classes[1]] = 1
    return y
