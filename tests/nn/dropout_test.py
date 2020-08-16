import numpy as np

import tor4
import tor4.nn as nn


def test_dropout_backward():
    a = tor4.tensor(
        data=[1, 2, 3, 4, 5, 6, 7, 8.0], dtype="float32", requires_grad=True
    )
    a_drop = nn.functional.dropout(a, p=0.8, training=True)
    a_drop.backward(tor4.tensor(data=[10, 10, 10.0, 10, 10, 10, 10, 10]))

    mask = a_drop.detach().numpy() == 0
    assert np.allclose(a_drop.detach().numpy()[mask], 0)
    assert np.allclose(a_drop.detach().numpy()[~mask], a.detach().numpy()[~mask] / 0.2)
    assert a_drop.requires_grad
    assert (a.grad.numpy()[mask] == 0).all()
    assert (a.grad.numpy()[~mask] == 10 / 0.2).all()


def test_dropout_eval_backward():
    a = tor4.tensor(
        data=[1, 2, 3, 4, 5, 6, 7, 8.0], dtype="float32", requires_grad=True
    )
    a_drop = nn.functional.dropout(a, p=0.8, training=False)
    a_drop.backward(tor4.tensor(data=[10, 10, 10.0, 10, 10, 10, 10, 10]))

    mask = a_drop.detach().numpy() != 0
    assert mask.all()
    assert a_drop.requires_grad
    assert a.grad.tolist() == [10, 10, 10, 10, 10, 10, 10, 10]


def test_dropout2d_backward():
    a = tor4.tensor(
        data=[[[1, 2, 3], [1, 2, 3.0]]], dtype="float32", requires_grad=True
    )
    a_drop = nn.functional.dropout2d(a, p=0.8, training=True)
    a_drop.backward(tor4.tensor(data=[[[2, 2, 2], [2, 2, 2.0]]]))

    mask = a_drop.detach().numpy() == 0
    if mask.any():
        assert mask[0].all(1).any()

    assert np.allclose(a_drop.detach().numpy()[mask], 0)
    assert np.allclose(a_drop.detach().numpy()[~mask], a.detach().numpy()[~mask] / 0.2)
    assert a_drop.requires_grad
    assert (a.grad.numpy()[a_drop.detach().numpy() == 0] == 0).all()
    assert (a.grad.numpy()[a_drop.detach().numpy() != 0] == 2 / 0.2).all()


def test_dropout2d_eval_backward():
    a = tor4.tensor(
        data=[[[1, 2, 3], [1, 2, 3.0]]], dtype="float32", requires_grad=True
    )
    a_drop = nn.functional.dropout2d(a, p=0.8, training=False)
    a_drop.backward(tor4.tensor(data=[[[2, 2, 2], [2, 2, 2.0]]]))

    mask = a_drop.detach().numpy() != 0
    assert mask.all()
    assert a_drop.requires_grad
    assert a.grad.tolist() == [[[2, 2, 2], [2, 2, 2.0]]]
