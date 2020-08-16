import numpy as np

import tor4
import tor4.nn as nn


def test_cross_entropy_backward():
    a = tor4.tensor([[1, 2, 3.0], [4, 5, 6]], requires_grad=True)
    b = tor4.tensor([1, 0])

    loss = nn.functional.cross_entropy(a, b)
    loss.backward()

    assert np.allclose(loss.item(), 1.9076, rtol=1e-4, atol=1e-4)
    assert np.allclose(
        a.grad.detach().numpy(),
        [[0.045, -0.3776, 0.3326], [-0.4550, 0.1224, 0.3326]],
        rtol=1e-4,
        atol=1e-4,
    )
