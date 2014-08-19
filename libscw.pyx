import numpy as np
cimport numpy as np
from scipy.stats import norm


__all__ = ['SCW1', 'SCW2']

ctypedef np.float64_t FLOAT_T
ctypedef np.int8_t INT_T


class BaseSCW(object): 
    def __init__(self, INT_T N_DIM, FLOAT_T C, FLOAT_T ETA):
        self.weights = np.zeros(N_DIM)
        self.covariance = np.ones(N_DIM)
        self.C = np.float64(C)
        self.cdf_values = self.calc_cdf_values(ETA)
    
    def sgn(self, np.ndarray[np.float64_t, ndim=1] x):
        cdef FLOAT_T t = np.dot(self.weights, x)
        return np.sign(t)

    def loss(self, np.ndarray[np.float64_t, ndim=1] x, FLOAT_T teacher):
        cdef FLOAT_T t = teacher * np.dot(self.weights, x)
        if(t >= 1):
            return 0
        return 1-t

    def calc_cdf_values(self, FLOAT_T ETA):
        cdef FLOAT_T phi, psi, zeta
        phi = norm.cdf(ETA)
        psi = 1 + np.power(phi, 2)/2
        zeta = 1 + np.power(phi, 2)
        return (phi, psi, zeta)

    def calc_confidence(self, np.ndarray[np.float64_t, ndim=1] x, 
                        FLOAT_T teacher):
        return np.dot(x, self.covariance*x)
    
    def calc_margin(self, np.ndarray[np.float64_t, ndim=1] x, FLOAT_T teacher):
        return teacher * np.dot(self.weights, x)
    
    def calc_alpha(self, np.ndarray[np.float64_t, ndim=1] x, FLOAT_T teacher):
        #calc in a child class
        pass

    def calc_beta(self, np.ndarray[np.float64_t, ndim=1] x, FLOAT_T teacher):
        cdef FLOAT_T phi, psi, zeta
        cdef FLOAT_T alpha, v, m 

        alpha = self.calc_alpha(x, teacher)
        v = self.calc_confidence(x, teacher)
        m = self.calc_margin(x, teacher) 
        phi, psi, zeta = self.cdf_values

        cdef FLOAT_T j, k, u
        j = -alpha * v * phi
        k = np.sqrt(np.power(alpha*v*phi, 2) + 4*v)
        u = np.power(j+k, 2) / 4
        return (alpha * phi) / (np.sqrt(u) + v*alpha*phi)

    def update_covariance(self, np.ndarray[np.float64_t, ndim=1] x, 
                          FLOAT_T teacher):
        cdef FLOAT_T beta
        cdef np.ndarray[np.float64_t, ndim=1] c
        beta = self.calc_beta(x, teacher)
        c = self.covariance
        self.covariance -= beta*c*c*x*x 

    def update_weights(self, np.ndarray[np.float64_t, ndim=1] x, 
                       FLOAT_T teacher):
        cdef FLOAT_T alpha
        alpha = self.calc_alpha(x, teacher)
        self.weights += alpha*teacher*self.covariance*x

    def update(self, np.ndarray[np.float64_t, ndim=1] x, FLOAT_T teacher):
        y = self.sgn(x)
        if(self.loss(x, teacher) > 0):
            self.update_weights(x, teacher)
            self.update_covariance(x, teacher)

    def train(self, np.ndarray[np.float64_t, ndim=2] X, 
              np.ndarray[np.float64_t, ndim=1] teachers):
        for x, teacher in zip(X, teachers):
            self.update(x, teacher)

    def fit(self, np.ndarray[np.float64_t, ndim=2] X, 
            np.ndarray[np.float64_t, ndim=1] teachers, int n_jobs):
        for i in range(n_jobs):
            self.train(X, teachers)
        return self.weights, self.covariance

    def weighted(self, np.ndarray[np.float64_t, ndim=2] X):
        rs = []
        for x in X:
            r = np.dot(x, self.weights)
            rs.append(r)
        return np.array(rs)
    
    def predict(self, np.ndarray[np.float64_t, ndim=2] X):
        labels = []
        for r in self.weighted(X):
            if(r > 0):
                labels.append(1)
            else:
                labels.append(-1)
        return np.array(labels)


class SCW1(BaseSCW):
    def calc_alpha(self, np.ndarray[np.float64_t, ndim=1] x, FLOAT_T teacher):
        v = self.calc_confidence(x, teacher)
        m = self.calc_margin(x, teacher) 
        phi, psi, zeta = self.cdf_values
        
        j = np.power(m, 2) * np.power(phi, 4) / 4
        k = v * zeta * np.power(phi, 2)
        t = (-m*psi + np.sqrt(j+k)) / (v*zeta)
        return min(self.C, max(0, t))


class SCW2(BaseSCW):
    def calc_alpha(self, np.ndarray[np.float64_t, ndim=1] x, FLOAT_T teacher):
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
