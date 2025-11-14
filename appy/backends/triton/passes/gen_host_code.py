import ast
import ast_comments as astc

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

def transform(tree: ast.Module, metadata):
    val_map = metadata['val_map']
    loop_name = metadata['loop_name']

    func = create_new_func(loop_name, val_map)
    # Make the current loop body code also be the body of the host function
    # This sets the basis for later tree transformations
    func.body = tree.body
    # Now make host_func the only content of the tree
    tree.body = [func]
    
    # Run codegen pass on the function
    from .host_passes import gen_data_movement
    from .host_passes import gen_kernel_launch

    func, h2d_map = gen_data_movement.transform(func, val_map)    
    func, replaced_loop = gen_kernel_launch.transform(func, h2d_map, metadata) 
    return tree, replaced_loop