import ast
import ast_comments as astc
from ...utils import load_module_from_str

def codegen(loop_source, loop_name, val_map, options):
    '''
    Returns a dynamically generated function from the loop source.
    '''
    tree = astc.parse(loop_source)    
    from .passes import sanity_check, parse_pragma, rewrite_range, block_loop
    sanity_check.visit(tree)
    pragma = parse_pragma.visit(tree)
    tree = rewrite_range.transform(tree)
    tree = block_loop.transform(tree, pragma)
    
    from .passes import gen_imports
    from .passes import gen_data_movement
    from .passes import gen_kernel_launch
    from .passes import gen_device_code
    
    metadata = {'loop_name': loop_name, 'val_map': val_map, 'options': options}
    tree, h2d_map = gen_data_movement.transform(tree, val_map)
    tree = gen_device_code.transform(tree, val_map, metadata)
    tree = gen_kernel_launch.transform(tree, val_map, h2d_map, metadata)        

    # Add imports at last!
    tree = gen_imports.transform(tree)
    ast.fix_missing_locations(tree)
    src = astc.unparse(tree)
    if options.get("dump_code"):
        print(f"--- Dumped code for loop {loop_name} ---")
        print(src)
        print(f"--- End of dumped code for loop {loop_name} ---")

    m = load_module_from_str(src)
    return getattr(m, loop_name)