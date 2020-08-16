from tor4 import tensor


def test_tensor_matmul():
    a = tensor(data=[1, 2, 3])
    b = tensor(data=[4, 5, 6])
    amb = a @ b

    assert amb.tolist() == 32
    assert not amb.requires_grad


def test_tensor_matmul2():
    a = tensor(data=[[1, 2, 3], [3, 2, 1]])
    b = tensor(data=[[1, 1], [1, 1], [1, 1]])
    amb = a @ b

    assert amb.tolist() == [
        [6, 6],
        [6, 6],
    ]
    assert not amb.requires_grad
    assert amb.shape == (2, 2)


def test_tensor_matmul2_backward():
    a = tensor(data=[[1, 2, 3], [3, 2, 1.0]], requires_grad=True)
    b = tensor(data=[[1, 1], [1, 1], [1, 1.0]], requires_grad=True)
    amb = a @ b
    amb.backward(tensor(data=[[1, 1], [1, 1]]))

    assert amb.tolist() == [
        [6, 6],
        [6, 6],
    ]
    assert amb.requires_grad
    assert a.grad.tolist() == [
        [2, 2, 2],
        [2, 2, 2],
    ]
    assert b.grad.tolist() == [
        [4, 4],
        [4, 4],
        [4, 4],
    ]


def test_tensor_matmul3_backward():
    a = tensor(
        data=[[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [7, 8, 9.0]]], requires_grad=True,
    )
    b = tensor(data=[[1, 1], [1, 1.0], [1, 2]], requires_grad=True)
    amb = a @ b
    amb.backward(tensor(data=[[[1, 1], [1, 2]], [[1, 1], [1, 1.0]]]))

    assert amb.tolist() == [[[6, 9], [6, 7]], [[15, 21], [24, 33]]]
    assert amb.requires_grad
    assert a.grad.tolist() == [[[2, 2, 3], [3, 3, 5]], [[2, 2, 3], [2, 2, 3]]]
    assert b.grad.tolist() == [
        [15, 18],
        [17, 19],
        [19, 20],
    ]


def test_tensor_matmul4_backward():
    a = tensor(
        data=[[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [7, 8, 9.0]]], requires_grad=True,
    )
    b = tensor(data=[[3, 1], [-1, 4.0], [1, 2]], requires_grad=True)
    amb = a @ b
    amb.backward(tensor(data=[[[-1, 1], [2, 3]], [[4, -5], [1, -6.0]]]))

    assert amb.tolist() == [[[4, 15], [8, 13]], [[13, 36], [22, 57]]]
    assert amb.requires_grad
    assert a.grad.tolist() == [
        [[-2, 5, 1], [9, 10, 8]],
        [[7, -24, -6], [-3, -25, -11]],
    ]
    assert b.grad.tolist() == [[28, -52], [30, -65], [32, -78]]
