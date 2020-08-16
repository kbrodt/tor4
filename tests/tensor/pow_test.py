from tor4 import tensor


def test_tensor_pow_scalar():
    a = tensor(data=[1, 2, 3])
    ap2 = a ** 2

    assert ap2.tolist() == [1, 4, 9]
    assert not ap2.requires_grad


def test_tensor_pow_scalar_backward():
    a = tensor(data=[1, 2, 3.0], requires_grad=True)
    ap2 = a ** 2
    ap2.backward(tensor(data=[1, 1, 1]))

    assert ap2.tolist() == [1, 4, 9]
    assert ap2.requires_grad
    assert a.requires_grad
    assert a.grad.tolist() == [2, 4, 6]
