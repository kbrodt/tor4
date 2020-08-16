import tor4


def test_no_grad():
    a = tor4.tensor(data=[1, 2, 3.0], requires_grad=True)
    b = tor4.tensor(data=[-1, 2, -3.0], requires_grad=True)

    with tor4.no_grad():
        c = (a + b) * b
        d = c.sum()

        assert a.requires_grad
        assert b.requires_grad
        assert not c.depends_on
        assert not d.depends_on
        assert not c.requires_grad
        assert not d.requires_grad
        with tor4.no_grad():
            cc = (a + b) * b
            dd = cc.sum()

            assert a.requires_grad
            assert b.requires_grad
            assert not cc.depends_on
            assert not dd.depends_on
            assert not cc.requires_grad
            assert not dd.requires_grad

        c = (a + b) * b
        d = c.sum()

        assert a.requires_grad
        assert b.requires_grad
        assert not c.depends_on
        assert not d.depends_on
        assert not c.requires_grad
        assert not d.requires_grad

    c = (a + b) * b
    d = c.sum()

    assert a.requires_grad
    assert b.requires_grad
    assert c.depends_on
    assert d.depends_on
    assert c.requires_grad
    assert d.requires_grad
