import os
import ast
import ast_comments as astc
from ...utils import load_module_from_str

code_cache = {}

def codegen(loop_source, loop_name, val_map, options):
    tree = astc.parse(loop_source)
    types = tuple([type(v) for v in val_map.values()])
    dtypes = tuple([v.dtype if hasattr(v, "dtype") else type(v) for v in val_map.values()])
    cache_key = (loop_source, types, dtypes, os.environ.get("TRITON_INTERPRET"))
    if cache_key in code_cache:            
        f, code_src = code_cache[cache_key]
    else:   
        # Do frontend trasformations
        from ...frontend import sanity_check
        #from .frontend import rewrite_prange
        from ...frontend import rewrite_range
        from ...frontend import rewrite_aug_assign
        from ...frontend import rewrite_tuple_assign
        from ...frontend import attach_pragma
        from ...frontend import reduction_detection

        sanity_check.visit(tree)
        tree = rewrite_tuple_assign.transform(tree)
        tree = rewrite_aug_assign.transform(tree)
        #tree = rewrite_prange.transform(tree)
        tree = rewrite_range.transform(tree)
        attach_pragma.visit(tree)
        tree = reduction_detection.transform(tree)

        from .passes import attach_shapes
        from .passes import block_loop
        from .passes import lower_array_op_to_loop

        attach_shapes.visit(tree, val_map)
        tree = lower_array_op_to_loop.transform(tree)
        tree = block_loop.transform(tree)    
        
        from .passes import gen_host_code
        from .passes import gen_device_code
        from .passes import gen_imports
        metadata = {'loop_name': loop_name, 'val_map': val_map, 'options': options}
        tree, replaced_loop = gen_host_code.transform(tree, metadata)
        tree = gen_device_code.transform(tree, replaced_loop, metadata)
        tree = gen_imports.transform(tree)

        ast.fix_missing_locations(tree)
        code_src = astc.unparse(tree)
        m = load_module_from_str(code_src)
        f = getattr(m, loop_name)    
        code_cache[cache_key] = (f, code_src)


        if options["dump_code"] or options["dump_code_to_file"]:
            try:
                import black
                code_src = black.format_str(code_src, mode=black.FileMode())
            except ImportError:
                pass            

        if options["dump_code"]:
            print(f"--- Dumped code for loop {loop_name} ---")            
            print(code_src)
            print(f"--- End of dumped code for loop {loop_name} ---")

        if options["dump_code_to_file"]:
            with open(options["dump_code_to_file"], "w") as f:
                f.write(code_src)

    return f, code_src