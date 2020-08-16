import numpy as np

from tor4 import tensor


def test_tensor_from_scalar():
    a = tensor(data=1)
    assert a.tolist() == 1
    assert not a.requires_grad
    assert a.is_leaf

    a = tensor(data=2.0, requires_grad=True)
    assert a.tolist() == 2.0
    assert a.requires_grad
    assert a.is_leaf


def test_tensor_from_list():
    a = tensor(data=[-1, 0, 1.0], requires_grad=True)
    assert a.tolist() == [-1, 0, 1]
    assert a.requires_grad
    assert a.is_leaf


def test_tensor_from_numpy():
    a_np = np.array([-1, 0, 1.0])
    a = tensor(data=a_np, requires_grad=True)
    assert a.tolist() == [-1, 0, 1]
    assert (a.detach().numpy() == a_np).all()
    assert a.requires_grad
    assert a.is_leaf
