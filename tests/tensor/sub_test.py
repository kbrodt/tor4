from tor4 import tensor


def test_tensor_sub():
    a = tensor(data=[1, 2, 3])
    am1 = a - 1

    assert am1.tolist() == [0, 1, 2]
    assert not am1.requires_grad


def test_tensor_sub_backward():
    a = tensor(data=[1, 2, 3.0], requires_grad=True)
    b = tensor(data=[3, 2, 1])

    amb = a - b
    amb.backward(tensor(data=[1, 1, 1]))

    assert amb.tolist() == [-2, 0, 2]
    assert amb.requires_grad
    assert a.requires_grad
    assert not b.requires_grad
    assert a.grad.tolist() == [1, 1, 1]


def test_tensor_neg():
    a = tensor(data=[1, 2, 3])
    ma = -a

    assert ma.tolist() == [-1, -2, -3]
    assert not ma.requires_grad


def test_tensor_neg_backward():
    a = tensor(data=[1, 2, 3.0], requires_grad=True)
    ma = -a
    ma.backward(tensor(data=[1, 1, 1]))

    assert ma.tolist() == [-1, -2, -3]
    assert ma.requires_grad
    assert a.requires_grad
    assert a.grad.tolist() == [-1, -1, -1]
