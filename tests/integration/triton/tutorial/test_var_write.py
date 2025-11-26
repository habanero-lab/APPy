import appy


def foo():
    a = 0
    for i in range(10):
        if i == 3:
            a = 1    
    return a


@appy.jit(dump_code=True)
def foo_appy():
    a = 0
    #pragma parallel for shared(a)
    for i in range(10):
        if i == 3:
            a = 1
    return a


def test():
    assert foo() == foo_appy()