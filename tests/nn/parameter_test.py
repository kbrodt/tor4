import tor4
import tor4.nn as nn


def test_parameter():
    a = nn.Parameter(data=tor4.arange(3, dtype="float32"))
    a.backward(tor4.ones(3))

    assert a.requires_grad
    assert a.grad.tolist() == [1, 1, 1]
