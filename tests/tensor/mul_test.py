from tor4 import tensor


def test_tensor_mul_with_scalar():
    a = tensor(data=[1, 2, 3])
    am2 = a * 2

    assert am2.tolist() == [2, 4, 6]
    assert not am2.requires_grad


def test_tensor_rmul_with_scalar():
    a = tensor(data=[1, 2, 3])
    am3 = 3 * a

    assert am3.tolist() == [3, 6, 9]
    assert not am3.requires_grad


def test_tensor_mul():
    a = tensor(data=[1, 2, 3])
    b = tensor(data=[-1, 3, 1])
    amb = a * b

    assert amb.tolist() == [-1, 6, 3]
    assert not amb.requires_grad


def test_tensor_mul_backward():
    a = tensor(data=[1, 2, 3])
    b = tensor(data=[-1, 3, 1.0], requires_grad=True)
    amb = a * b
    amb.backward(tensor([1, 2, 3]))

    assert amb.tolist() == [-1, 6, 3]
    assert not a.requires_grad
    assert b.requires_grad
    assert amb.requires_grad
    assert a.grad is None
    assert b.grad.tolist() == [1, 4, 9]


def test_tensor_rmul_backward():
    a = tensor(data=[1, 2, 3.0], requires_grad=True)
    b = tensor(data=[-1, 3, 1])
    amb = a * b
    amb.backward(tensor([3, 2, 1]))

    assert amb.tolist() == [-1, 6, 3]
    assert a.requires_grad
    assert not b.requires_grad
    assert amb.requires_grad
    assert a.grad.tolist() == [-3, 6, 1]
    assert b.grad is None


def test_tensor_imul_backward():
    a = tensor(data=[1, 2, 3.0], requires_grad=True)
    b = tensor(data=[-1, 3, 1])
    try:
        a *= b
        assert False
    except RuntimeError:
        assert True


def test_tensor_mul_broadcast_backward():
    a = tensor(data=[[1, 2, 3], [1, 1, 2]])
    b = tensor(data=[-1, 3, 1.0], requires_grad=True)
    amb = a * b
    amb.backward(tensor([[1, 1, 1], [1, 1, 1]]))

    assert amb.tolist() == [[-1, 6, 3], [-1, 3, 2]]
    assert not a.requires_grad
    assert b.requires_grad
    assert amb.requires_grad
    assert a.grad is None
    assert b.grad.tolist() == [2, 3, 5]


def test_tensor_mul_broadcast2_backward():
    a = tensor(data=[[1, 2, 3], [1, 1, 2]])
    b = tensor(data=[[-1, 3, 1.0]], requires_grad=True)
    amb = a * b
    amb.backward(tensor([[1, 1, 1], [1, 1, 1]]))

    assert amb.tolist() == [[-1, 6, 3], [-1, 3, 2]]
    assert not a.requires_grad
    assert b.requires_grad
    assert amb.requires_grad
    assert a.grad is None
    assert b.grad.tolist() == [[2, 3, 5]]


def test_tensor_mul_broadcast3_backward():
    a = tensor(data=[[[1, 2, 3], [1, 1, 2]], [[1, 2, 3], [1, 1, 2]]])
    b = tensor(data=[[1], [0.0]], requires_grad=True)
    amb = a * b
    amb.backward(tensor([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]))

    assert amb.tolist() == [[[1, 2, 3], [0, 0, 0]], [[1, 2, 3], [0, 0, 0]]]
    assert not a.requires_grad
    assert b.requires_grad
    assert amb.requires_grad
    assert a.grad is None
    assert b.grad.tolist() == [[12], [8]]
