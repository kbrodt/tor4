import tor4


def test_sigmoid_backward():
    a = tor4.tensor(data=[0.0, 0, 0], requires_grad=True)
    a_sigmoid = a.sigmoid()
    a_sigmoid.backward(tor4.tensor(data=[1, 1, 1]))

    assert a_sigmoid.tolist() == [0.5, 0.5, 0.5]
    assert a_sigmoid.requires_grad
    assert a.grad.tolist() == [0.25, 0.25, 0.25]
