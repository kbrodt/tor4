import tor4
import tor4.nn as nn


def test_linear():
    linear = nn.Linear(in_features=3, out_features=10, bias=True)
    x = tor4.ones(2, 3)
    logits = linear(x)
    loss = logits.sum()
    loss.backward()

    assert not x.requires_grad
    assert linear.weight.requires_grad
    assert linear.bias.requires_grad
    assert linear.weight.shape == (10, 3)
    assert linear.bias.shape == (10,)
    assert len(list(linear.parameters())) == 2

    assert logits.shape == x.shape[:-1] + (10,)
    assert logits.requires_grad

    # assert linear.weight.grad.tolist() == (x.data.T @ logits.grtor4.data).tolist()

    linear.zero_grad()
    assert linear.weight.grad.tolist() == tor4.zeros(10, 3).tolist()
    assert linear.bias.grad.tolist() == tor4.zeros(10).tolist()
