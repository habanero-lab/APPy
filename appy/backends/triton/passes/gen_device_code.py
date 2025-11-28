import ast
from .utils import is_numpy_array, is_torch_tensor

def create_new_func(loop_name, val_map):
    '''
    Create a new function which contains the host code. The name of the function
    is simply the loop_name, and all variables in val_map become arguments.
    '''
    func = ast.FunctionDef(
        name=loop_name,
        args=ast.arguments(
            posonlyargs=[],      # required in 3.8+
            args=[ast.arg(arg=val, annotation=None) for val in val_map],
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

def add_stride_params(func, val_map):
    extra_args = []
    for var, val in val_map.items():
        if is_numpy_array(val) or is_torch_tensor(val):
            # Add a stride parameter for each dimension except the last one
            # So basically only ndim >= 2 will have stride parameters
            for d in range(len(val.shape) - 1):
                extra_args.append(ast.arg(arg=f'{var}_stride_{d}', annotation=None))

    func.args.args.extend(extra_args)

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
    func = create_new_func('_' + loop_name, val_map) 

    add_stride_params(func, val_map)
    add_triton_decorator(func)

    # Add the replaced loop into the kernel function body, which will be transformed
    tree.body.append(func)
    func.body.append(replaced_loop)
    
    # Run codegen pass on the function
    from .device_passes import rewrite_par_reduce_to_atomics
    from .device_passes import remove_loop_head
    from .device_passes import rewrite_vidx
    from .device_passes import attach_masks
    from .device_passes import lower_subscripts
    from .device_passes import lower_constants
    from .device_passes import rewrite_np_calls
    from .device_passes import apply_mask_to_reduction
    from .device_passes import rewrite_ternary

    attach_masks.visit(func)
    func = rewrite_par_reduce_to_atomics.transform(func)
    func = remove_loop_head.transform(func)
    func = apply_mask_to_reduction.transform(func)
    func = rewrite_vidx.transform(func)

    func = lower_subscripts.transform(func)
    func = lower_constants.transform(func)
    # ast.fix_missing_locations(tree)
    # print(ast.unparse(tree))
    # exit(0)
    func = rewrite_np_calls.transform(func)
    func = rewrite_ternary.transform(func)
    return tree