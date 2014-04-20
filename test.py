import numpy as np

import scw

from matplotlib import pyplot

def plot(X, teachers, weights):
    red_t = []
    blue_t = []
    for x, teacher in zip(X, teachers):
        if(teacher > 0.):
            red_t.append(x)
        else:
            blue_t.append(x)
    red_t = np.array(red_t)
    blue_t = np.array(blue_t)
    
    red_c = []
    blue_c = []
    for x in zip(X):
        if(np.dot(x, weights) > 0.):
            red_c.append(x[0])
        else:
            blue_c.append(x[0]) 
    red_c = np.array(red_c)
    blue_c = np.array(blue_c)
    
    pyplot.subplot(211)
    pyplot.plot(red_t.T[0], red_t.T[1], 'ro', 
                blue_t.T[0], blue_t.T[1], 'bs')
    
    pyplot.subplot(212)
    pyplot.plot(red_c.T[0], red_c.T[1], 'ro', 
                blue_c.T[0], blue_c.T[1], 'bs')
    pyplot.show()


class Data(object):
    def __init__(self):
        self.X = []
        self.t = []

    def add(self, p=None, n=None):
        if(p is not None):
            self.X += p
            self.t += [1] * len(p)

        if(n is not None):
            self.X += n
            self.t += [-1] * len(n)
        
    def get(self):
        return np.array(self.X), np.array(self.t)


if(__name__ == '__main__'):
    N = 20

    data = Data() 
    data.add(p=[[0, 1], [4, 4], [1, 2]], 
             n=[[-2, 1], [0, -1], [-1, 0], [2, -4]])
    X, t = data.get()
    
    scw = scw.SCW2(len(X[0]), C=10.0, ETA=1.0)
    weights, covariance = scw.train(X, t)
    print("weights:{}".format(weights))
    plot(X, t, weights)

    data.add(n=[[-1, 3], [1, -4]])
    X, t = data.get()
    weights, covariance = scw.train(X, t)
    print("weights:{}".format(weights))
    plot(X, t, weights)
