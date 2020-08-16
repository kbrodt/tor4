import numpy as np

import tor4
import tor4.nn as nn


def test_quadratic():
    w = tor4.tensor(data=[1, 2], dtype=tor4.float32)
    w0 = tor4.tensor(data=np.random.rand(2), dtype=tor4.float32, requires_grad=True)

    n_iters = 50
    lr = 1
    eps = 1e-15
    for _ in range(n_iters):
        loss = nn.functional.mse_loss(w0, w)
        if abs(loss.item()) < eps:
            break

        loss.backward()

        with tor4.no_grad():
            w0 -= lr * w0.grad

            w0.grad.zero_()

    assert np.allclose(
        w.numpy(), w0.detach().numpy()
    ), f"Something wrong with learning. Optimal point {w.numpy()}, found {w0.numpy()}"


def test_quadratic2():
    w = tor4.tensor(data=[1, 2, 3], dtype=tor4.float32)
    w0 = nn.Parameter(tor4.randn(3))

    n_iters = 50
    lr = 1
    eps = 1e-15
    opt = tor4.optim.SGD([w0], lr=lr)
    for _ in range(n_iters):
        loss = nn.functional.mse_loss(w0, w)
        if abs(loss.item()) < eps:
            break

        opt.zero_grad()
        loss.backward()
        opt.step()

    assert np.allclose(
        w.numpy(), w0.detach().numpy()
    ), f"Something wrong with learning. Optimal point {w.numpy()}, found {w0.numpy()}"


def test_linear():
    weight = nn.Parameter(tor4.randn(10, 3))
    bias = nn.Parameter(tor4.randn(10))
    model = nn.Linear(3, 10)
    model.weight = weight
    model.bias = bias

    X = tor4.randn(100, 3)
    y = model(X)

    model = nn.Linear(3, 10)

    n_iters = 100
    lr = 1
    eps = 1e-15
    opt = tor4.optim.SGD(model.parameters(), lr=lr)
    for _ in range(n_iters):
        logits = model(X)
        loss = nn.functional.mse_loss(logits, y)
        if abs(loss.item()) < eps:
            break

        opt.zero_grad()
        loss.backward()
        opt.step()

    assert np.allclose(
        weight.detach().numpy(), model.weight.detach().numpy(), rtol=1e-4, atol=1e-4,
    ), f"Something wrong with learning. Optimal weights {weight.numpy()}, found {model.weight.numpy()}"
    assert np.allclose(
        bias.detach().numpy(), model.bias.detach().numpy(), rtol=1e-4, atol=1e-4,
    ), f"Something wrong with learning. Optimal bias {bias.numpy()}, found {model.bias.numpy()}"
