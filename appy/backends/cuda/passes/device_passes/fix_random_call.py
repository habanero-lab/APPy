import ast


def transform(node):
    '''
    Inserts the thread index `tid` as the seed argument to random() calls.
    '''
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and \
                isinstance(child.func, ast.Name) and child.func.id == 'random':
            child.func.id = 'appy_random'
            child.args.append(ast.Name(id='tid', ctx=ast.Load()))
    return node
