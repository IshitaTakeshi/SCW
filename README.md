Exact Soft Confidence-Weighted Learning
=======================================

##The algorithm
This is an online supervised learning algorithm which utilizes all the four salient properties:

* Large margin training
* Confidence weighting
* Capability to handle non-separable data
* Adaptive margin

The paper is [here](http://icml.cc/2012/papers/86.pdf).

SCW has 2 formulations of its algorithm which are SCW-I and SCW-II.
They can be accessed like below.

```
scw.SCW1(C, ETA)
scw.SCW2(C, ETA)
```

C and ETA are hyperparameters.

## Installation
### Dependent packages
* numpy
* scipy

### Install via pip
```
pip3 install scw
```

##Usage

```
from scw import SCW1, SCW2

scw = SCW1(C=1.0, ETA=1.0)
scw.fit(X, y)
y_pred = scw.perdict(X)
```

`X` and `y` are 2-dimensional and 1-dimensional array respectively.  
`X` is a set of data vectors. Each row of `X` represents a feature vector.  
`y` is a set of labels corresponding with `X`.  

## Note
1. This package performs only binary classification, not multiclass classification.
2. Training labels must be 1 or -1. No other labels allowed.
