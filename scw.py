import numpy as np
import libscw

class BaseSCW(object):
    def fit(self, X, teachers, n_jobs=1):
        X = np.array(X).astype(np.float64)
        teachers = np.array(teachers).astype(np.float64)
        if not(isinstance(n_jobs, int)):
            raise TypeError("n_jobs must be int")
        self.scw.fit(X, teachers, n_jobs)

    def predict(self, X):
        X = np.array(X).astype(np.float64)
        return self.scw.predict(X) 


class SCW1(BaseSCW):
    def __init__(self, N_DIM, C, ETA):
        self.scw = libscw.SCW1(N_DIM, C, ETA)


class SCW2(BaseSCW):
    def __init__(self, N_DIM, C, ETA):
        self.scw = libscw.SCW2(N_DIM, C, ETA)
