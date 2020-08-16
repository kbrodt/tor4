import tor4
import tor4.nn as nn


def test_mse_backward():
    inputs = tor4.tensor(data=[1.0, 2, 3], requires_grad=True)
    targets = tor4.tensor(data=[2, 3, 2])

    mse_nn = nn.functional.mse_loss(inputs, targets)
    mse = ((inputs - targets) ** 2).mean()
    mse2 = nn.functional.mse_loss_slow(inputs, targets)

    mse_nn.backward()

    assert mse_nn.item() == mse.item() == mse2.item() == 1
    assert mse_nn.requires_grad
    assert inputs.grad.tolist() == [-2 / 3, -2 / 3, 2 / 3]
