import os
import ast
import ast_comments as astc
from pathlib import Path
from ...utils import load_module_from_str

code_cache = {}

def codegen(loop_source, loop_name, val_map, options):
    tree = astc.parse(loop_source)
    types = tuple([type(v) for v in val_map.values()])
    dtypes = tuple([v.dtype if hasattr(v, "dtype") else type(v) for v in val_map.values()])
    shapes = tuple([v.shape[1:] if hasattr(v, "shape") else None for v in val_map.values()])
    cache_key = (loop_source, types, dtypes, shapes)
    if cache_key in code_cache:            
        f, code_src = code_cache[cache_key]
    else:   
        # Do frontend transformation
        from ...frontend import rewrite_aug_assign
        from ...frontend import rewrite_tuple_assign

        tree = rewrite_tuple_assign.transform(tree)
        tree = rewrite_aug_assign.transform(tree)

        # Metal specific codegen
        from .passes import attach_types
        from .passes import gen_host_code
        from .passes import gen_device_code

        attach_types.visit(tree, val_map)

        tree, replaced_loop = gen_host_code.transform(tree, {'loop_name': loop_name, 'val_map': val_map})
        tree = gen_device_code.transform(tree, replaced_loop, loop_name, val_map)

        # code_src = Path(f"{Path(__file__).parent}/sample_kernels/gelu.py").read_text()
        # m = load_module_from_str(code_src)
        # f = getattr(m, loop_name)
        ast.fix_missing_locations(tree)
        code_src = astc.unparse(tree)
      
        if options['dump_code']:
            print(f"--- Dumped code for loop {loop_name} ---")          
            print(code_src.replace("\\n", "\n"))
            print(f"--- End of dumped code for loop {loop_name} ---")

        m = load_module_from_str(code_src)
        f = getattr(m, loop_name)

        code_cache[cache_key] = f, code_src

    return f, code_src