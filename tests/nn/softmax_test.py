import numpy as np

import tor4
import tor4.nn as nn


def test_softmax():
    a = tor4.tensor(data=[0, 0, 0.0])
    a_sm = nn.functional.softmax(a, dim=0)

    assert not a_sm.requires_grad
    assert a_sm.tolist() == [1 / 3, 1 / 3, 1 / 3]


def test_softmax2():
    a = tor4.tensor(data=[[0, 0, 0], [0, 0, 0.0]])
    a_sm0 = nn.functional.softmax(a, dim=0)
    a_sm1 = nn.functional.softmax(a, dim=1)

    assert not a_sm0.requires_grad
    assert a_sm0.tolist() == [[1 / 2, 1 / 2, 1 / 2], [1 / 2, 1 / 2, 1 / 2]]
    assert not a_sm1.requires_grad
    assert a_sm1.tolist() == [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]


def test_softmax_backward():
    a = tor4.tensor(data=[0, 0, 0.0], requires_grad=True)
    a_sm = nn.functional.softmax(a, dim=-1)
    a_sm.backward(tor4.tensor([1, 1, 1.0]))

    assert a_sm.requires_grad
    assert a_sm.tolist() == [1 / 3, 1 / 3, 1 / 3]
    assert a.grad.tolist() == [0, 0, 0]


def test_softmax_backward2():
    a = tor4.tensor(data=[0, 0, 0.0], requires_grad=True)
    a_sm = nn.functional.softmax(a, dim=-1)
    a_sm.backward(tor4.tensor([0, 1, -1.0]))

    assert a_sm.requires_grad
    assert a_sm.tolist() == [1 / 3, 1 / 3, 1 / 3]
    assert a.grad.tolist() == [0, 1 / 3, -1 / 3]


def test_softmax2d_backward():
    a = tor4.tensor(data=[[0, 1, -1.0], [1, -2, 3]], requires_grad=True)
    a_sm = nn.functional.softmax(a, dim=-1)
    a_sm.backward(tor4.tensor([[1, 1, 1.0], [1, 1, 1]]))

    assert a_sm.requires_grad
    assert np.allclose(
        a_sm.tolist(),
        [[0.2447, 0.6652, 0.09], [0.1185, 0.0059, 0.8756]],
        atol=1e-4,
        rtol=1e-4,
    )
    assert np.allclose(a.grad.tolist(), [[0, 0, 0], [0, 0, 0]])


def test_softmax2d_backward2():
    a = tor4.tensor(data=[[0, 1, -1.0], [1, -2, 3]], requires_grad=True)
    a_sm = nn.functional.softmax(a, dim=0)
    a_sm.backward(tor4.tensor([[1, 1, 1.0], [1, 1, 1]]))

    assert a_sm.requires_grad
    assert np.allclose(
        a_sm.tolist(),
        [[0.2689, 0.9526, 0.018], [0.7311, 0.0474, 0.982]],
        atol=1e-4,
        rtol=1e-4,
    )
    assert np.allclose(a.grad.tolist(), [[0, 0, 0], [0, 0, 0]])


def test_softmax2d_backward3():
    a = tor4.tensor(data=[[0, 1, -1.0], [1, -2, 3]], requires_grad=True)
    a_sm = nn.functional.softmax(a, dim=-1)
    a_sm.backward(tor4.tensor([[0, -1, 1.0], [2, 0, -1]]))

    assert a_sm.requires_grad
    assert np.allclose(
        a_sm.tolist(),
        [[0.2447, 0.6652, 0.09], [0.1185, 0.0059, 0.8756]],
        atol=1e-4,
        rtol=1e-4,
    )
    assert np.allclose(
        a.grad.tolist(),
        [[0.1408, -0.2826, 0.1418], [0.3127, 0.0038, -0.3164]],
        atol=1e-4,
        rtol=1e-4,
    )


def test_softmax2d_backward4():
    a = tor4.tensor(data=[[0, 1, -1.0], [1, -2, 3]], requires_grad=True)
    a_sm = nn.functional.softmax(a, dim=0)
    a_sm.backward(tor4.tensor([[-5, 3, 0.0], [0, 0, 1]]))

    assert a_sm.requires_grad
    assert np.allclose(
        a_sm.tolist(),
        [[0.2689, 0.9526, 0.018], [0.7311, 0.0474, 0.982]],
        atol=1e-4,
        rtol=1e-4,
    )
    assert np.allclose(
        a.grad.tolist(),
        [[-0.9831, 0.1355, -0.0177], [0.9831, -0.1355, 0.0177]],
        atol=1e-4,
        rtol=1e-4,
    )


def test_softmax3d_backward():
    a = tor4.tensor(
        data=[[[0, 1, -1.0], [1, -2, 3]], [[1, 4, -2], [0, 0, -3]]], requires_grad=True,
    )
    a_sm = nn.functional.softmax(a, dim=1)
    a_sm.backward(tor4.tensor([[[-5, 3, 0.0], [0, 0, 1]], [[3, 0, -3], [1, 2, 3]]]))

    assert a_sm.requires_grad
    assert np.allclose(
        a_sm.tolist(),
        [
            [[0.2689, 0.9526, 0.018], [0.7311, 0.0474, 0.982]],
            [[0.7311, 0.982, 0.7311], [0.2689, 0.018, 0.2689]],
        ],
        atol=1e-4,
        rtol=1e-4,
    )
    assert np.allclose(
        a.grad.tolist(),
        [
            [[-0.9831, 0.1355, -0.0177], [0.9831, -0.1355, 0.0177]],
            [[0.3932, -0.0353, -1.1797], [-0.3932, 0.0353, 1.1797]],
        ],
        atol=1e-4,
        rtol=1e-4,
    )
