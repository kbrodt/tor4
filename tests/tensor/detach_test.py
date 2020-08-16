import tor4


def test_detach():
    a = tor4.tensor(data=[1, 2, 3.0], requires_grad=True)
    a_sum = a.sum()
    a_sum.backward()

    assert a_sum.requires_grad
    assert a.requires_grad
    assert a.grad.tolist() == [1, 1, 1]

    b = a.detach()

    assert not b.requires_grad
    assert b.grad is None
    assert a.requires_grad
    assert a.grad.tolist() == [1, 1, 1]
