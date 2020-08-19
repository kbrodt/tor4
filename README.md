# tor4: PyTorch on NumPy

The purpose of this repository is to understand of how [PyTorch](https://pytorch.org/)
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
    - 'SGD' - stochastic gradient descent

# Requirements

- Python 3.6+
- [`Poetry`](https://python-poetry.org/)

# Installation

```bash
poetry install --extras 'dev'
```

# Testing

```bash
poetry run pytest
```

# Usage

You can use it as `torch`: the API is almost the same.

Here an example on MNIST dataset. To train using simple linear layers run:

```bash
poetry run python -m examples.mnist
```

There is also a model with convolutional layers:

```bash
USE_CONV= poetry run python -m examples.mnist
```

# TODOs

- [ ] Boolean operations
- [ ] `bias` and padding support to `Conv2d`
- [ ] Pooling layers
- [ ] `Conv1d'

# Usefull links

- This library is inspired by [Joel's livecoding of an autograd library](https://www.youtube.com/watch?v=RxmBukb-Om4&list=PLeDtc0GP5ICldMkRg-DkhpFX1rRBNHTCs)
