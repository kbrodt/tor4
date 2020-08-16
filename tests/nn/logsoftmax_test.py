import numpy as np

import tor4
import tor4.nn as nn


def test_logsoftmax_backward():
    a = tor4.tensor([0.0, 0, 0], requires_grad=True)
    lsm = nn.functional.log_softmax(a, dim=-1)
    lsm.backward(tor4.tensor([1, 1, 2.0]))

    assert np.allclose(lsm.tolist(), np.log([1 / 3] * 3))
    assert lsm.requires_grad
    assert np.allclose(a.grad.tolist(), [-1 / 3, -1 / 3, 2 / 3])


def test_logsoftmax_backward1():
    a = tor4.tensor([1.0, 2, 3], requires_grad=True)
    lsm = nn.functional.log_softmax(a, dim=-1)
    lsm.backward(tor4.tensor([1, 1, 2.0]))

    assert np.allclose(lsm.tolist(), [-2.4076, -1.4076, -0.4076], rtol=1e-4, atol=1e-4)
    assert lsm.requires_grad
    assert np.allclose(a.grad.tolist(), [0.6399, 0.0211, -0.661], atol=1e-4, rtol=1e-4)


def test_logsoftmax_backward2():
    a = tor4.tensor([[1, 2, -3], [10.0, 0, -1]], requires_grad=True)
    lsm = nn.functional.log_softmax(a, dim=-1)
    lsm.backward(tor4.tensor([[2, 4, -1], [1, 1, 2.0]]))

    assert np.allclose(
        lsm.tolist(),
        [[-1.3182, -0.31818, -5.3182], [0, -10, -11]],
        rtol=1e-4,
        atol=1e-4,
    )
    assert lsm.requires_grad
    assert np.allclose(
        a.grad.tolist(),
        [[0.6619, 0.3626, -1.0245], [-2.9998, 0.9998, 1.9999]],
        rtol=1e-4,
        atol=1e-4,
    )


def test_logsoftmax_backward3():
    a = tor4.tensor([[1, 2, -3], [10.0, 0, -1]], requires_grad=True)
    lsm = nn.functional.log_softmax(a, dim=0)
    lsm.backward(tor4.tensor([[2, 4, -1], [1, 1, 2.0]]))

    assert np.allclose(
        lsm.tolist(),
        [[-9.0001, -0.12693, -2.1269], [-0.0001, -2.1269, -0.12693]],
        rtol=1e-4,
        atol=1e-4,
    )
    assert lsm.requires_grad
    assert np.allclose(
        a.grad.tolist(),
        [[1.9996, -0.404, -1.1192], [-1.9996, 0.4040, 1.1192]],
        rtol=1e-4,
        atol=1e-4,
    )
