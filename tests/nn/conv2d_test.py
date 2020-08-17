import numpy as np

import tor4
import tor4.nn as nn


def test_conv2d_backward():
    a = tor4.tensor([[[[0, 1, 2], [3, 4, 5.0], [6, 7, 8]]]], requires_grad=True)
    w = tor4.tensor([[[[1, 0], [1, 1]]], [[[0, 1], [0, 1.0]]]], requires_grad=True)

    aw = nn.functional.conv2d(a, w)
    aw.backward(tor4.tensor([[[[1, -2], [2, 1]], [[3, 2.0], [-1, 1]]]]))

    assert aw.tolist() == [[[[7, 10], [16, 19]], [[5, 7], [11, 13]]]]
    assert np.allclose(
        a.grad.tolist(), [[[[1, 1, 2], [3, 2, 1], [2, 2, 2]]]], rtol=1e-4, atol=1e-4,
    )
    assert np.allclose(
        w.grad.tolist(),
        [[[[8, 10], [14, 16]]], [[[3, 8], [18, 23]]]],
        rtol=1e-4,
        atol=1e-4,
    )


def test_conv2d_backward2():
    a = tor4.tensor([[[[0, 1, 2], [3, 4, 5.0]]]], requires_grad=True)
    w = tor4.tensor([[[[1, 0], [1, 1]]], [[[0, 1], [0, 1.0]]]], requires_grad=True)

    aw = nn.functional.conv2d(a, w, stride=2)
    aw.backward(tor4.tensor([[[[1]], [[2.0]]]]))

    assert aw.tolist() == [[[[7]], [[5]]]]
    assert np.allclose(
        a.grad.tolist(), [[[[1, 2, 0], [1, 3, 0]]]], rtol=1e-4, atol=1e-4
    )
    assert np.allclose(
        w.grad.tolist(), [[[[0, 1], [3, 4]]], [[[0, 2], [6, 8]]]], rtol=1e-4, atol=1e-4,
    )


def test_conv2d_backward3():
    a = tor4.tensor([[[[0, 1, 2], [3, 4, 5.0]]]])
    w = tor4.tensor([[[[1, 0], [1, 1]]], [[[0, 1], [0, 1.0]]]], requires_grad=True)

    try:
        nn.functional.conv2d(a, w, dilation=2)
        raise AssertionError()
    except RuntimeError:
        assert True


def test_conv2d_backward31():
    a = tor4.arange(9, dtype=tor4.float32, requires_grad=True)
    b = a.view(1, 1, 3, 3)
    w = tor4.tensor([[[[1, 0], [1, 1]]], [[[0, 1], [0, 1.0]]]], requires_grad=True)

    aw = nn.functional.conv2d(b, w, dilation=2)
    aw.backward(tor4.tensor([[[[2]], [[1.0]]]]))

    assert aw.tolist() == [[[[14]], [[10]]]]
    assert np.allclose(a.grad.tolist(), [2, 0, 1, 0, 0, 0, 2, 0, 3])
    assert np.allclose(w.grad.tolist(), [[[[0, 4], [12, 16]]], [[[0, 2], [6, 8]]]])


def test_conv2d_backward4():
    a = tor4.tensor(
        [
            [
                [[0, 1, 2], [3, 4, 5.0]],
                [[0, 1, 2], [3, 4, 5.0]],
                [[0, 1, 2], [3, 4, 5.0]],
            ]
        ],
        requires_grad=True,
    )
    w = tor4.tensor(
        [
            [[[1, 0], [1, 1]], [[1, 0], [1, 1]], [[1, 0], [1, 1]]],
            [[[0, 1], [0, 1.0]], [[0, 1], [0, 1.0]], [[0, 1], [0, 1.0]]],
        ],
        requires_grad=True,
    )

    aw = nn.functional.conv2d(a, w)
    aw.backward(tor4.tensor([[[[-1, 3]], [[1.0, 2]]]]))

    assert aw.tolist() == [[[[21, 30]], [[15, 21]]]]
    assert np.allclose(
        a.grad.tolist(),
        [
            [
                [[-1, 4, 2], [-1, 3, 5.0]],
                [[-1, 4, 2], [-1, 3, 5.0]],
                [[-1, 4, 2], [-1, 3, 5.0]],
            ]
        ],
        rtol=1e-4,
        atol=1e-4,
    )
    assert np.allclose(
        w.grad.tolist(),
        [
            [[[3, 5], [9, 11]], [[3, 5], [9, 11]], [[3, 5], [9, 11]]],
            [[[2, 5], [11, 14]], [[2, 5], [11, 14]], [[2, 5], [11, 14]]],
        ],
        rtol=1e-4,
        atol=1e-4,
    )


def test_conv2d_backward5():
    a = tor4.tensor(
        [
            [
                [[0, 1, 2], [3, 4, 5.0]],
                [[0, 1, 2], [3, 4, 5.0]],
                [[0, 1, 2], [3, 4, 5.0]],
            ]
        ],
        requires_grad=True,
    )
    w = tor4.tensor(
        [[[[1, 0], [1, 1]]], [[[0, 1], [0, 1.0]]], [[[1, 0], [1, 1]]]],
        requires_grad=True,
    )

    aw = nn.functional.conv2d(a, w, groups=3)
    aw.backward(tor4.tensor([[[[2, -1]], [[1, 3]], [[3, -2.0]]]]))

    assert aw.tolist() == [[[[7, 10]], [[5, 7]], [[7, 10]]]]
    assert np.allclose(
        a.grad.sum(1).tolist(), [[[5, -2, 3], [5, 3, 0]]], rtol=1e-4, atol=1e-4
    )
    # assert np.allclose(a.grad.tolist(), [[[[2, -1, 0], [2, 1, -1]], [[0, 1, 3], [0, 1, 3.]], [[3, -2, 0], [3, 1, -2.]]]], rtol=1e-4, atol=1e-4)
    assert np.allclose(
        w.grad.tolist(),
        [[[[-1, 0], [2, 3]]], [[[3, 7], [15, 19]]], [[[-2, -1], [1, 2]]]],
        rtol=1e-4,
        atol=1e-4,
    )
