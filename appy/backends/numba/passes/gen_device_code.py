import ast

def create_new_func(loop_name, val_map):
    '''
    Create a new function which contains the host code. The name of the function
    is simply the loop_name, and all variables in val_map become arguments.
    '''
    func = ast.FunctionDef(
        name=loop_name,
        args=ast.arguments(
            posonlyargs=[],      # required in 3.8+
            args=[ast.arg(arg=var, annotation=None) for var in val_map],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[],
        decorator_list=[],
    )
    return func

def add_numba_decorator(func: ast.FunctionDef):
    # func.decorator_list.append(ast.Attribute(
    #     value=ast.Name(id='numba', ctx=ast.Load()),
    #     attr='njit',
    #     ctx=ast.Load(),
    # ))
    func.decorator_list.append(ast.parse('numba.njit(parallel=True)').body[0].value)

def replace_appy_prange(loop):
    for node in ast.walk(loop):
        if isinstance(node, ast.For):
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute) and \
            ast.unparse(node.iter.func.value) == 'appy':
                node.iter.func.value = ast.Name(id='numba', ctx=ast.Load())    

def transform(tree, replaced_loop, loop_name, val_map):
    device_func = create_new_func('_kernel', val_map)
    add_numba_decorator(device_func)
    replace_appy_prange(replaced_loop)
    device_func.body.append(replaced_loop)
    tree.body.append(device_func)
    return tree
