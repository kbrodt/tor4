from tor4 import tensor


def test_tesnor_div_scalar():
    a = tensor(data=[1, 2, 3])
    ad2 = a / 2

    assert ad2.tolist() == [1 / 2, 1, 3 / 2]
    assert not ad2.requires_grad


def test_tesnor_div_scalar_backward():
    a = tensor(data=[1.0, 2.0, 3.0], requires_grad=True)
    ad2 = a / 2
    ad2.backward(tensor(data=[1.0, 1.0, 1.0]))

    assert ad2.tolist() == [1 / 2, 1, 3 / 2]
    assert ad2.requires_grad
    assert a.requires_grad
    assert a.grad.tolist() == [1 / 2, 1 / 2, 1 / 2]


def test_tesnor_rdiv_scalar_backward():
    a = tensor(data=[1.0, 2.0, -3.0], requires_grad=True)
    ad2 = 1 / a
    ad2.backward(tensor(data=[1.0, 1.0, 1.0]))

    assert ad2.tolist() == [1, 1 / 2, -1 / 3]
    assert ad2.requires_grad
    assert a.requires_grad
    assert a.grad.tolist() == [-1, -1 / 4, -1 / 9]


def test_tesnor_idiv_scalar():
    a = tensor(data=[1.0, 2.0, 3.0], requires_grad=True)

    try:
        a /= 2.0
        raise AssertionError()
    except RuntimeError:
        assert True


def test_tesnor_div_backward():
    a = tensor(data=[2.0, 9, -2.0], requires_grad=True)
    b = tensor(data=[2, 3, 4.0], requires_grad=True)
    adb = a / b
    adb.backward(tensor(data=[1.0, 1.0, 1.0]))

    assert adb.tolist() == [1, 3, -0.5]
    assert adb.requires_grad
    assert a.requires_grad
    assert a.grad.tolist() == [1 / 2, 1 / 3, 1 / 4]
    assert b.requires_grad
    assert b.grad.tolist() == [-1 / 2, -1, 1 / 8]
