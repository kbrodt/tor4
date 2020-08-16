from tor4 import tensor


def test_tesnor_neg():
    a = tensor(data=[1, 2, 3])
    na = -a

    assert na.tolist() == [-1, -2, -3]
    assert not na.requires_grad


def test_tesnor_neg_backward():
    a = tensor(data=[1, 2, 3.0], requires_grad=True)
    na = -a
    na.backward(tensor([1, 1, 1]))

    assert na.tolist() == [-1, -2, -3]
    assert na.requires_grad
    assert a.grad.tolist() == [-1, -1, -1]
