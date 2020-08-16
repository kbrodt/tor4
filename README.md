# tor4: PyTroch on NumPy

The purpose of this repository is to understand of how [PyTroch](https://pytorch.org/)
library is implemented under the hood and what an
[automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) is.

We will based on three whales:

- [`Python`](https://www.python.org/)
- [`NumPy`](https://numpy.org/) and [`SciPy`](https://www.scipy.org/)
- Basic knowledge of [Calculus](https://en.wikipedia.org/wiki/Calculus),
  namely [differential calculus](https://en.wikipedia.org/wiki/Differential_calculus).
  [Linear algebra](https://en.wikipedia.org/wiki/Linear_algebra),
  namely [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication).

This repository provides with:

* Pointwise operations
    - elementary arithmetic operations (addition, substraction, multiplication, division, etc.)
    - elementary functions (`pow`, `exp`, `log`, etc.)
    - special functions (`sigmoid`, `softmax`, etc.)
* Reduction operations
    - `sum`, `mean`, `max`
* BLAS  and LAPACK operation
    - batch matrix-matrix product
* Indexing and slicing operations

and theirs gradient calculations to perform backward pass.

Also there are some basic building blocks for neural networks:

* Activations
    - `ReLU`
* Linear
    - `Linear`
* Convolutional
    - `Conv2d`
* Dropout
    - `Dropout1d`
    - `Dropout2d`
* Loss functions
    - MSE - mean squared error
    - BCE - binary cross entropy
    - XENT - categorical cross entropy
* Optimizer
    - SGD - stochastic gradient descent

# Requirements

- Python 3.6+
- [`Poetry`](https://python-poetry.org/)

# Installation

```bash
poetry install
```

# Testing

```bash
poetry run pytest
```

# Usage

You can use it as mini-`torch`: the interface is kept as much as possible the same.

Here an example on MNIST dataset. To train using simple linear layers run:

```bash
poetry run python -m examples.mnist
```

There is also a model with convolutional layers:

```bash
USE_CONV= poetry run python -m examples.mnist
```

# TODOs

- [ ] Add boolean operations
- [ ] Add pooling layers
- [ ] Add `bias` and padding support to `Conv2d`

# Usefull links

- This library is inspired by [Joel's livecoding of an autograd library](https://www.youtube.com/watch?v=RxmBukb-Om4&list=PLeDtc0GP5ICldMkRg-DkhpFX1rRBNHTCs)
- [Linear backprop](http://cs231n.stanford.edu/handouts/linear-backprop.pdf)
- [2002, Computational Methods in Decision-Making, Economics and Finance](https://www.springer.com/gp/book/9781402008399), page 297, chapter 15, Automatic Differentiation for Computation Finance
