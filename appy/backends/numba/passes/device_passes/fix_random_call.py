import ast


def transform(node):
    '''
    This pass inserts an argument `grid_id` to the random function calls, which should have no arguments.
    '''

    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == 'random':
            child.args.append(ast.Name(id='grid_id', ctx=ast.Load()))
    return node