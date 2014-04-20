import numpy as np
from scipy.stats import norm

class SCW(object):
    def __init__(self, ndim, C=1.0, ETA=1.0):
        self.weights = np.zeros(ndim)
        self.covariance = np.ones(ndim)
        self.C = np.float64(C)
        self.cdf_values = self.calc_cdf_values(ETA)
    
    def sgn(self, x):
        t = np.dot(self.weights, x)
        if(t > 0):
            return 1
        if(t == 0):
            return 0
        if(t < 0):
            return -1

    def loss(self, x, teacher):
        t = teacher*np.dot(self.weights, x)
        if(t >= 1):
            return 0
        return 1-t

    def calc_cdf_values(self, ETA):
        phi = norm.cdf(ETA)
        psi = 1 + np.power(phi, 2)/2
        zeta = 1 + np.power(phi, 2)
        return (phi, psi, zeta)

    def calc_confidence(self, x, teacher):
        return np.dot(x, self.covariance*x)
    
    def calc_margin(self, x, teacher):
        return teacher*np.dot(self.weights, x)
    
    def calc_alpha(self, x, teacher):
        #calc in a child class
        pass

    def calc_beta(self, x, teacher):
        alpha = self.calc_alpha(x, teacher)
        v = self.calc_confidence(x, teacher)
        m = self.calc_margin(x, teacher) 
        phi, psi, zeta = self.cdf_values

        j = -alpha * v * phi
        k = np.sqrt(np.power(alpha*v*phi, 2) + 4*v)
        u = np.power(j+k, 2) / 4
        return (alpha * phi) / (np.sqrt(u) + v*alpha*phi)

    def update_covariance(self, x, teacher):
        beta = self.calc_beta(x, teacher)
        c = self.covariance
        self.covariance -= beta*c*c*x*x 

    def update_weights(self, x, teacher):
        alpha = self.calc_alpha(x, teacher)
        self.weights += alpha*teacher*self.covariance*x

    def update(self, x, teacher):
        y = self.sgn(x)
        if(self.loss(x, teacher) > 0):
            self.update_weights(x, teacher)
            self.update_covariance(x, teacher)

    def train(self, X, teachers):
        for x, teacher in zip(X, teachers):
            self.update(x, teacher)
        return self.weights, self.covariance


class SCW1(SCW):
    def calc_alpha(self, x, teacher):
        v = self.calc_confidence(x, teacher)
        m = self.calc_margin(x, teacher) 
        phi, psi, zeta = self.cdf_values
        
        j = np.power(m, 2) * np.power(phi, 4) / 4
        k = v * zeta * np.power(phi, 2)
        t = (-m*psi + np.sqrt(j+k)) / (v*zeta)
        return min(self.C, max(0, t))


class SCW2(SCW):
    def calc_alpha(self, x, teacher):
        v = self.calc_confidence(x, teacher)
        m = self.calc_margin(x, teacher) 
        phi, psi, zeta = self.cdf_values
        
        n = v+1/self.C
        a = np.power(phi*m*v, 2)
        b = 4*n*v * (n+v*np.power(phi, 2))
        gamma = phi * np.sqrt(a+b)
        
        c = -(2*m*n + m*v*np.power(phi, 2))
        d = np.power(n, 2) + n*v*np.power(phi, 2)
        t = (c+gamma)/(2*d)
        return max(0, t)
