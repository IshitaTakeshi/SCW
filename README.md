#Exact Soft Confidence-Weighted Learning

##The explanation of the algorithm
This is an online supervised learning algorithm.  
This learning method enjoys all the four salient properties:

* Large margin training
* Confidence weighting
* Capability to handle non-separable data
* Adaptive margin 

The paper is [here](http://icml.cc/2012/papers/86.pdf).

There are 2 kinds of implementations presented in the paper, which served as 

```
scw.SCW1(data_dimension, C, ETA)
scw.SCW2(data_dimension, C, ETA)
```

in the code. C and ETA are hyperparameters.

##Usage

```
from scw import SCW1, SCW2

scw = SCW1(C=1.0, ETA=1.0)
weights, covariance = scw.fit(training_data, teachers)
results = scw.perdict(test_data)
```

`teachers` is 1-dimensional and `training_data` and `test_data` are 2-dimensional array.
