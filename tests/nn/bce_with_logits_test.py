import numpy as np
from scipy import special

import tor4
import tor4.nn as nn


def test_bce_with_logits_backward():
    a_np = np.array([1.0, 0, -1])
    a = tor4.tensor(data=a_np, requires_grad=True)
    b_np = np.array([0.0, 1, 0])
    b = tor4.tensor(data=b_np, requires_grad=True)

    loss = nn.functional.binary_cross_entropy_with_logits(a, b)
    loss.backward()

    sigmoid_a_np = special.expit(a_np)
    bce = -np.log([1 - sigmoid_a_np[0], sigmoid_a_np[1], 1 - sigmoid_a_np[2]]).mean()
    shape = np.prod(a_np.shape)
    grad = (special.expit(a_np) - b_np) / shape

    assert loss.requires_grad
    assert np.allclose(loss.item(), bce)
    assert np.allclose(a.grad.numpy(), grad)
    assert np.allclose(b.grad.numpy(), -a_np / shape)


def test_bce_with_logits_backward2():
    a_np = np.array([[1.0, 0, -1], [-1, 0, 1]])
    a = tor4.tensor(data=a_np, requires_grad=True)
    b_np = np.array([[0.0, 1, 0], [1, 0, 1]])
    b = tor4.tensor(data=b_np, requires_grad=True)

    loss = nn.functional.binary_cross_entropy_with_logits(a, b)
    loss.backward()

    sigmoid_a_np = special.expit(a_np)
    bce = -np.log(
        [
            [1 - sigmoid_a_np[0, 0], sigmoid_a_np[0, 1], 1 - sigmoid_a_np[0, 2]],
            [sigmoid_a_np[1, 0], 1 - sigmoid_a_np[1, 1], sigmoid_a_np[1, 2]],
        ]
    ).mean()
    shape = np.prod(a_np.shape)
    grad = (special.expit(a_np) - b_np) / shape

    assert loss.requires_grad
    assert np.allclose(loss.item(), bce)
    assert np.allclose(a.grad.numpy(), grad)
    assert np.allclose(b.grad.numpy(), -a_np / shape)
