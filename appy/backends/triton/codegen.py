import ast
import ast_comments as astc

def codegen(tree, loop_name, val_map, options):
    '''
    Returns a dynamically generated function from the loop source.
    '''
    from .passes import block_loop
    from .passes import lower_array_op_to_loop

    tree = lower_array_op_to_loop.transform(tree, val_map)
    tree = block_loop.transform(tree)    
    
    from .passes import gen_host_code
    from .passes import gen_device_code
    from .passes import gen_imports
    metadata = {'loop_name': loop_name, 'val_map': val_map, 'options': options}
    tree, replaced_loop = gen_host_code.transform(tree, metadata)
    tree = gen_device_code.transform(tree, replaced_loop, metadata)
    tree = gen_imports.transform(tree)

    ast.fix_missing_locations(tree)
    src = astc.unparse(tree)
    return src