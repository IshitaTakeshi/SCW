import time

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from scw import SCW1, SCW2


def generate_dataset():
    X, y = datasets.make_classification(n_samples=5000, n_features=10)
    data = train_test_split(X, y)
    training = data[0], data[2]
    test = data[1], data[3]
    return training, test


def profile(model, model_name, training, test):
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


training, test = generate_dataset()

profile(SCW1(), "SCW1", training, test)
profile(LinearSVC(), "LinearSVC", training, test)
