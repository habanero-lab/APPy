import ast

def create_new_func(name):
    return ast.FunctionDef(
        name=name,
        args=ast.arguments(
            args=[],
            vararg=None,
            kwarg=None,
            defaults=[],
        ),
        body=[],
        decorator_list=[],
    )

def set_func_params(func, val_map):
    func.args.args = [ast.arg(arg=val, annotation=None) for val in val_map]

def add_triton_decorator(func: ast.FunctionDef):
    func.decorator_list.append(ast.Attribute(
        value=ast.Name(id='triton', ctx=ast.Load()),
        attr='jit',
        ctx=ast.Load(),
    ))
    
def transform(tree: ast.Module, replaced_loop: ast.For, metadata):
    val_map = metadata['val_map']
    loop_name = metadata['loop_name']

    # Create function and set its parameters etc
    func = create_new_func('_' + loop_name)
    set_func_params(func, val_map)
    add_triton_decorator(func)

    # Add the replaced loop into the kernel function body, which will be transformed
    tree.body.append(func)
    func.body.append(replaced_loop)
    
    # Run codegen pass on the function
    from .device_passes import remove_loop_head
    from .device_passes import rewrite_vidx
    from .device_passes import attach_masks
    from .device_passes import lower_subscripts
    from .device_passes import lower_constants

    func = remove_loop_head.transform(func)
    
    attach_masks.visit(func)
    func = rewrite_vidx.transform(func)

    # ast.fix_missing_locations(tree)
    # print(ast.unparse(tree))
    # exit(0)

    func = lower_subscripts.transform(func)
    func = lower_constants.transform(func)
    return tree