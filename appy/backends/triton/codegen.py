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
    
    from .passes import gen_host_code
    from .passes import gen_device_code
    from .passes import gen_imports
    metadata = {'loop_name': loop_name, 'val_map': val_map, 'options': options}
    tree, replaced_loop = gen_host_code.transform(tree, metadata)
    tree = gen_device_code.transform(tree, replaced_loop, metadata)
    tree = gen_imports.transform(tree)

    ast.fix_missing_locations(tree)
    src = astc.unparse(tree)
    if options.get("dump_code"):
        print(f"--- Dumped code for loop {loop_name} ---")
        print(src)
        print(f"--- End of dumped code for loop {loop_name} ---")

    m = load_module_from_str(src)
    return getattr(m, loop_name)