import time
import profile
import pstats

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

import utils
import datasets
from scw import SCW1, SCW2



def accuracy_and_time(model, model_name, training, test):
    print("Model: {}".format(model_name))

    X, y = training

    t1 = time.time()
    model.fit(X, y)
    t2 = time.time()
    print("    Spent time for training :  {}".format(t2-t1))

    X, y_true = test
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_pred, y_true)
    print("    Accuracy :  {}\n".format(accuracy))


def run_profile(model, model_name, X, y):
    filename = model_name+".def"
    profile.runctx("for i in range(100): model.fit(X, y)",
                   globals(), locals(), filename)

    #p = pstats.Stats(filename)
    #p.print_stats()



X, y = datasets.make_classification()
training, test = utils.train_test_split(X, y)
accuracy_and_time(SCW1(), "SCW1", training, test)
accuracy_and_time(SCW2(), "SCW2", training, test)
accuracy_and_time(LinearSVC(), "LinearSVC", training, test)

X, y = datasets.load_digits()
run_profile(SCW1(), "SCW1", X, y)
run_profile(SCW2(), "SCW2", X, y)
