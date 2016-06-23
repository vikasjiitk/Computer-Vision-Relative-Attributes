Learning to Rank
======

A simple implementation of algorithms of learning to rank. Pairwise (RankNet) approach.

## Requirements
[tqdm](https://github.com/noamraph/tqdm)

[matplotlib v1.5.1](http://matplotlib.org/)

[numpy v1.10.1](http://www.numpy.org/)

[scipy]()

[chainer v1.5.1](http://chainer.org/)

[scikit-learn](http://scikit-learn.org/stable/)

and some basic packages.


## RankNet
### Pairwise comparison of rank

The original paper was written by Chris Burges et al., "Learning to Rank using Gradient Descent." (available at http://research.microsoft.com/en-us/um/people/cburges/papers/ICML_ranking.pdf)

### Usage

Import and initialize

```
import RankNet
Model = RankNet.RankNet()
```

Fitting (automatically do training and validation)

```
Model.fit(X, y)
```

Possible options and defaults:

```
batchsize=100, n_iter=5000, n_units1=512, n_units2=128, tv_ratio=0.95, optimizerAlgorithm="Adam", savefigName="result.pdf", savemodelName="RankNet.model"
```

```n_units1``` and ```n_units2=128``` are the number of nodes in hidden layer 1 and 2 in the neural net.

```tv_ratio``` is the ratio of the data amounts between training and validation. 

## Author

If you have any troubles or questions, please contact [shiba24](https://github.com/shiba24).

March, 2016

