import tor4
from tor4 import tensor


def test_tensor_slice():
    a = tensor(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    a1 = a[1]

    assert a1.tolist() == [4, 5, 6]
    assert not a1.requires_grad


def test_tensor_slice2():
    a = tensor(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    a1 = a[None]

    assert a1.shape == (1, 3, 3)
    assert a1.tolist() == [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]

    assert not a1.requires_grad


def test_tensor_slice_backward():
    a = tensor(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9.0]], requires_grad=True)

    a1 = a[..., 1]
    a1.backward(tensor(data=[1, 1, 1]))

    assert a1.tolist() == [2, 5, 8]
    assert a1.requires_grad
    assert a.grad.tolist() == [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]


def test_tensor_slice2_backward():
    a = tensor(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9.0]], requires_grad=True)

    a1 = a[..., [2, 0]]
    a1.backward(tensor(data=[[1, 1], [1, 1], [1, 1]]))

    assert a1.tolist() == [[3, 1], [6, 4], [9, 7]]
    assert a1.requires_grad
    assert a.grad.tolist() == [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]


def test_tensor_slice3_backward():
    a = tensor(data=[[1.0, 2, 3], [4, 5, 6], [7, 8, 9.0]], requires_grad=True)

    a1 = a[None]
    a1.backward(tor4.ones(1, 3, 3))

    assert a1.tolist() == [[[1, 2, 3], [4, 5, 6], [7, 8, 9.0]]]

    assert a1.requires_grad
