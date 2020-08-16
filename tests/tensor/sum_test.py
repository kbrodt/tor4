from tor4 import tensor


def test_tensor_sum():
    a = tensor(data=[-1, 1, 2])
    a_sum = a.sum()

    assert a_sum.tolist() == 2
    assert not a_sum.requires_grad


def test_tensor_sum_backward():
    a = tensor(data=[-1, 1, 2.0], requires_grad=True)
    a_sum = a.sum()
    a_sum.backward()

    assert a_sum.tolist() == 2
    assert a_sum.requires_grad
    assert a.grad.tolist() == [1, 1, 1]


def test_tensor_sum_backward2():
    a = tensor(data=[-1, 1, 2.0], requires_grad=True)
    a_sum = a.sum()
    a_sum.backward(tensor(3))

    assert a_sum.tolist() == 2
    assert a_sum.requires_grad
    assert a.grad.tolist() == [3, 3, 3]


def test_tensor_sum1_backward():
    a = tensor(data=[[-1, 1, 2], [1, 2, 3.0]], requires_grad=True)
    a_sum = a.sum(dim=1)
    a_sum.backward(tensor(data=[2, 3]))

    assert a_sum.tolist() == [2, 6]
    assert a_sum.requires_grad
    assert a.grad.tolist() == [[2, 2, 2], [3, 3, 3]]


def test_tensor_sum2_backward():
    a = tensor(data=[[[-1], [1], [2]], [[1], [2], [3.0]]], requires_grad=True)
    a_sum = a.sum(dim=1)
    a_sum.backward(tensor(data=[[2], [3]]))

    assert a_sum.tolist() == [[2], [6]]
    assert a_sum.requires_grad
    assert a.grad.tolist() == [[[2], [2], [2]], [[3], [3], [3]]]


def test_tensor_sum3_backward():
    a = tensor(data=[[[-1], [1], [2]], [[1], [2], [3.0]]], requires_grad=True)
    a_sum = a.sum()
    a_sum.backward()

    assert a_sum.tolist() == 8
    assert a_sum.requires_grad
    assert a.grad.tolist() == [[[1], [1], [1]], [[1], [1], [1]]]


def test_tensor_sum4_backward():
    a = tensor(data=[[[-1], [1], [2]], [[1], [2], [3.0]]], requires_grad=True)
    a_sum = a.sum(dim=(1, 0))
    a_sum.backward()

    assert a_sum.tolist() == [8]
    assert a_sum.requires_grad
    assert a.grad.tolist() == [[[1], [1], [1]], [[1], [1], [1]]]


def test_tensor_sum_keepdim1_backward():
    a = tensor(data=[[-1, 1, 2], [1, 2, 3.0]], requires_grad=True)
    a_sum = a.sum(dim=1, keepdim=True)
    a_sum.backward(tensor(data=[[2], [3]]))

    assert a_sum.tolist() == [[2], [6]]
    assert a_sum.requires_grad
    assert a.grad.tolist() == [[2, 2, 2], [3, 3, 3]]


def test_tensor_sum_keepdim2_backward():
    a = tensor(data=[[[-1], [1], [2]], [[1], [2], [3.0]]], requires_grad=True)
    a_sum = a.sum(dim=1, keepdim=True)
    a_sum.backward(tensor(data=[[[2]], [[3]]]))

    assert a_sum.tolist() == [[[2]], [[6]]]
    assert a_sum.requires_grad
    assert a.grad.tolist() == [[[2], [2], [2]], [[3], [3], [3]]]


def test_tensor_sum_keepdim3_backward():
    a = tensor(data=[[[-1], [1], [2]], [[1], [2], [3.0]]], requires_grad=True)
    a_sum = a.sum()
    a_sum.backward()

    assert a_sum.tolist() == 8
    assert a_sum.requires_grad
    assert a.grad.tolist() == [[[1], [1], [1]], [[1], [1], [1]]]


def test_tensor_sum_keepdim4_backward():
    a = tensor(data=[[[-1], [1], [2]], [[1], [2], [3.0]]], requires_grad=True)
    a_sum = a.sum(dim=(1, 0), keepdim=True)
    a_sum.backward()

    assert a_sum.tolist() == [[[8]]]
    assert a_sum.requires_grad
    assert a.grad.tolist() == [[[1], [1], [1]], [[1], [1], [1]]]
