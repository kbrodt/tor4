import tor4


def test_max_backward():
    a = tor4.tensor(data=[1, 3, -1.0], requires_grad=True)
    a_max = a.max()
    a_max.backward()

    assert a_max.tolist() == 3
    assert a_max.requires_grad
    assert a.requires_grad
    assert a.grad.tolist() == [0, 1, 0]


def test_max2_backward():
    a = tor4.tensor(data=[[1, 3, -1], [-3, 4, 1.0]], requires_grad=True)
    a_max = a.max()
    a_max.backward()

    assert a_max.tolist() == 4
    assert a_max.requires_grad
    assert a.requires_grad
    assert a.grad.tolist() == [[0, 0, 0], [0, 1, 0]]


def test_max3_backward():
    a = tor4.tensor(data=[[1, 3, -1], [-3, 4, 1.0]], requires_grad=True)
    a_max, arg_max = a.max(dim=0)
    a_max.backward(tor4.tensor([1, 1, 1]))

    assert a_max.tolist() == [1, 4, 1]
    assert a_max.requires_grad
    assert arg_max.tolist() == [0, 1, 1]
    assert not arg_max.requires_grad
    assert a.grad.tolist() == [[1, 0, 0], [0, 1, 1]]
    assert a.requires_grad


def test_max4_backward():
    a = tor4.tensor(data=[[10, 3, -1], [-3, 4, 10.0]], requires_grad=True)
    a_max, arg_max = a.max(dim=1, keepdim=True)
    a_max.backward(tor4.tensor([[1], [1]]))

    assert a_max.tolist() == [[10], [10]]
    assert a_max.requires_grad
    assert arg_max.tolist() == [[0], [2]]
    assert not arg_max.requires_grad
    assert a.grad.tolist() == [[1, 0, 0], [0, 0, 1]]
    assert a.requires_grad


def test_max5_backward():
    a = tor4.tensor(
        data=[[[[10, 3, -1], [-3, 4, 10]], [[10, 3, -1], [-3, 4, 10.0]]]],
        requires_grad=True,
    )
    a_max, _ = a.max(dim=-1)
    a_max2, _ = a_max.max(dim=-1)

    loss = a_max2.mean()

    loss.backward()

    assert a_max.tolist() == [[[10, 10], [10, 10]]]
    assert a_max2.tolist() == [[10, 10]]
    assert loss.item() == 10
    assert a.grad.tolist() == [[[[0.5, 0, 0], [0, 0, 0]], [[0.5, 0, 0], [0, 0, 0]]]]
