import tor4


def test_retain_graph():
    a = tor4.tensor([1.0, 2], requires_grad=True)
    asum = a.sum()
    asum.backward(retain_graph=True)

    assert a.grad.tolist() == [1, 1]

    asum.backward()
    assert a.grad.tolist() == [2, 2]

    asum.backward()
    assert a.grad.tolist() == [2, 2]


def test_retain_graph2():
    a = tor4.tensor([1.0, 2], requires_grad=True)
    b = tor4.tensor([2.0, 3], requires_grad=True)
    apb = a + b
    apb2 = apb ** 2
    apbp3 = apb + 3
    res = apb2 + apbp3
    loss = res.sum()
    loss.backward()

    assert a.grad.tolist() == [7, 11]
    assert b.grad.tolist() == [7, 11]
