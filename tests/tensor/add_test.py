import numpy as np

from tor4 import tensor


def test_tensor_add_with_scalar():
    a = tensor([1, 2, 3])
    ap1 = a + 1

    assert a.is_leaf
    assert ap1.tolist() == [2, 3, 4]
    assert not ap1.requires_grad
    assert ap1.is_leaf


def test_tensor_radd_with_scalar():
    a = tensor([1, 2, 3])
    ap2 = 2 + a

    assert a.is_leaf
    assert ap2.tolist() == [3, 4, 5]
    assert not ap2.requires_grad
    assert ap2.is_leaf


def test_tensor_add():
    a = tensor([1, 2, 3])
    b = tensor([4, 5, 6])
    apb = a + b

    assert apb.tolist() == [5, 7, 9]
    assert not apb.requires_grad
    assert apb.is_leaf


def test_tensor_add_backward():
    a = tensor([1, 2, 3])
    b = tensor([-1, 0, 1.0], requires_grad=True)
    apb = a + b
    apb.backward(tensor([1, 1, 1]))

    assert a.is_leaf
    assert b.is_leaf
    assert not apb.is_leaf
    assert apb.tolist() == [0, 2, 4]
    assert not a.requires_grad
    assert b.requires_grad
    assert apb.requires_grad
    assert a.grad is None
    assert b.grad.tolist() == [1, 1, 1]


def test_tensor_radd_backward():
    a = tensor([1, 2, 3.0], requires_grad=True)
    b = tensor([-1, 0, 1])
    apb = a + b
    apb.backward(tensor([1, 1, 2]))

    assert apb.tolist() == [0, 2, 4]
    assert a.requires_grad
    assert not b.requires_grad
    assert apb.requires_grad
    assert a.grad.tolist() == [1, 1, 2]
    assert b.grad is None


def test_tensor_iadd_backward():
    a = tensor([1, 2, 3.0], requires_grad=True)
    b = tensor([-1, 0, 1])

    try:
        a += b
        raise AssertionError()
    except RuntimeError:
        assert True


def test_tensor_add_broadcast_backward():
    a = tensor([[1, 2, 3], [4, 5, 6]])
    b = tensor([-1, 0, 1.0], requires_grad=True)
    apb = a + b
    apb.backward(tensor([[1, 1, 1], [1, 1, 1]]))

    assert apb.tolist() == [
        [0, 2, 4],
        [3, 5, 7],
    ]
    assert not a.requires_grad
    assert b.requires_grad
    assert apb.requires_grad
    assert a.grad is None
    assert b.grad.tolist() == [2, 2, 2]


def test_tensor_add_broadcast2_backward():
    a = tensor([[1, 2, 3], [4, 5, 6]])
    b = tensor([[-1, 0, 1.0]], requires_grad=True)
    apb = a + b
    apb.backward(tensor([[1, 1, 1], [1, 1, 1]]))

    assert apb.tolist() == [
        [0, 2, 4],
        [3, 5, 7],
    ]
    assert not a.requires_grad
    assert b.requires_grad
    assert apb.requires_grad
    assert a.grad is None
    assert b.grad.tolist() == [[2, 2, 2]]


def test_tensor_add_broadcast3_backward():
    a = tensor(
        [
            [[1, 2, 3, 3], [4, 5, 6, 6], [4, 3, 2, 1]],
            [[1, 2, 3, 3], [4, 5, 6, 6], [4, 3, 2, 1]],
        ]
    )
    b = tensor([[-1], [0], [1.0]], requires_grad=True)
    apb = a + b
    apb.backward(tensor(np.ones((2, 3, 4), dtype=a.dtype)))

    assert apb.tolist() == [
        [[0, 1, 2, 2], [4, 5, 6, 6], [5, 4, 3, 2]],
        [[0, 1, 2, 2], [4, 5, 6, 6], [5, 4, 3, 2]],
    ]
    assert not a.requires_grad
    assert b.requires_grad
    assert apb.requires_grad
    assert a.grad is None
    assert b.grad.tolist() == [[8], [8], [8]]
