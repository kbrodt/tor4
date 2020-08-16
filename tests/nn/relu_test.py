import tor4
import tor4.nn as nn


def test_relu_backward():
    a = tor4.tensor(data=[-1, -2, 0, 4.0], requires_grad=True)
    a_relu = nn.functional.relu(a)
    a_relu.backward(tor4.tensor(data=[1, 1, 1, 1]))

    assert a_relu.requires_grad
    assert a_relu.tolist() == [0, 0, 0, 4]
    assert a.grad.tolist() == [0, 0, 0, 1]
