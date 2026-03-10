import ast
import ast_comments as astc


def create_new_func(loop_name, val_map):
    '''
    Create a new function containing the host code.  All variables in
    val_map become positional arguments.  (No `device` arg needed for CUDA —
    pycuda.autoinit handles device selection globally.)
    '''
    func = ast.FunctionDef(
        name=loop_name,
        args=ast.arguments(
            posonlyargs=[],
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


def transform(tree: ast.Module, metadata):
    val_map = metadata['val_map']
    loop_name = metadata['loop_name']

    func = create_new_func(loop_name, val_map)
    func.body = tree.body
    tree.body = [func]

    tree.body.insert(0, ast.Import(names=[ast.alias(name='numpy', asname='np')]))

    from .host_passes import gen_kernel_launch
    func, replaced_loop = gen_kernel_launch.transform(
        func, loop_name, val_map, metadata.get('use_simd', False))

    return tree, replaced_loop
