import math

import numpy as np
from numpy import matlib
from scipy.stats import norm

__all__ = ['SCW1', 'SCW2']


class BaseSCW(object):
    def __init__(self, C=1.0, ETA=1.0):
        self.weights = None
        self.covariance = None
        self.C = np.float64(C)
        self.cdf_values = self.calc_cdf_values(ETA)
        self.has_fitted = False

    def loss(self, x, label):
        t = label * self.weights.T*x
        if t >= 1:
            return 0
        return 1-t

    def calc_cdf_values(self, ETA):
        phi = norm.cdf(ETA)
        psi = 1 + pow(phi, 2)/2
        zeta = 1 + pow(phi, 2)
        return (phi, psi, zeta)

    def calc_confidence(self, x):
        c = x.T*self.covariance*x
        return c.item(0)  # get a scalar value

    def calc_margin(self, x, label):
        y = self.weights.T*x
        return label*y.item(0)

    def calc_alpha(self, x, label):
        # calc in a child class
        pass

    def calc_beta(self, x, label):
        alpha = self.calc_alpha(x, label)
        v = self.calc_confidence(x)
        m = self.calc_margin(x, label)
        phi, psi, zeta = self.cdf_values

        j = -alpha * v * phi
        k = math.sqrt(pow(alpha*v*phi, 2) + 4*v)
        u = pow(j+k, 2) / 4
        return (alpha * phi) / (math.sqrt(u) + v*alpha*phi)

    def update_covariance(self, x, label):
        beta = self.calc_beta(x, label)
        c = self.covariance
        self.covariance -= beta*c.T*x*x.T*c

    def update_weights(self, x, label):
        alpha = self.calc_alpha(x, label)
        self.weights += alpha*label*self.covariance*x

    def update(self, x, label):
        if label != 1 and label != -1:
            raise ValueError("Data label must be 1 or -1.")

        if self.loss(x, label) > 0:
            self.update_weights(x, label)
            self.update_covariance(x, label)

    def fit_(self, X, labels):
        for x, label in zip(X, labels):
            x = x.T  # regard x as a vector
            self.update(x, label)

    def fit(self, X, labels):
        X = matlib.matrix(X)

        n_dim = X.shape[1]
        if np.ndim(X) != 2:
            raise ValueError("Estimator expencs 2 dim array.")

        if not self.has_fitted:
            self.weights = np.matlib.zeros((n_dim, 1))
            self.covariance = matlib.eye(n_dim)
            self.has_fitted = True

        self.fit_(X, labels)
        return self

    def predict(self, X):
        labels = []
        for x in X:
            r = np.dot(x, self.weights)
            if r > 0:
                labels.append(1)
            else:
                labels.append(-1)
        return labels


class SCW1(BaseSCW):
    def calc_alpha(self, x, label):
        v = self.calc_confidence(x)
        m = self.calc_margin(x, label)
        phi, psi, zeta = self.cdf_values

        j = pow(m, 2) * pow(phi, 4) / 4
        k = v * zeta * pow(phi, 2)
        t = (-m*psi + math.sqrt(j+k)) / (v*zeta)
        return min(self.C, max(0, t))


class SCW2(BaseSCW):
    def calc_alpha(self, x, label):
        v = self.calc_confidence(x)
        m = self.calc_margin(x, label)
        phi, psi, zeta = self.cdf_values

        n = v+1/self.C
        a = pow(phi*m*v, 2)
        b = 4*n*v * (n + v*pow(phi, 2))
        gamma = phi * math.sqrt(a+b)

        c = -(2*m*n + m*v*pow(phi, 2))
        d = pow(n, 2) + n*v*pow(phi, 2)
        t = (c+gamma)/(2*d)
        return max(0, t)
