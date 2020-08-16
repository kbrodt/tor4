import tor4


def test_view_backward():
    a = tor4.tensor([1.0, 2, 3, 4, 5, 6, 7, 8, 9], requires_grad=True)
    b = a.view(-1, 3)
    b.backward(tor4.tensor([[1, 1, 1], [2, 2, 2], [3, 4, 5.0]]))

    assert b.shape == (3, 3)
    assert b.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert a.grad.tolist() == [1, 1, 1, 2, 2, 2, 3, 4, 5]


def test_view_backward2():
    a = tor4.tensor([[[1, 2], [3, 4], [5, 6.0]]], requires_grad=True)
    b = a.view(2, 3)
    b.backward(tor4.tensor([[0, 1, 0], [6, 3, 1.0]]))

    assert b.shape == (2, 3)
    assert b.tolist() == [[1, 2, 3], [4, 5, 6.0]]
    assert a.grad.tolist() == [[[0, 1], [0, 6], [3, 1]]]
