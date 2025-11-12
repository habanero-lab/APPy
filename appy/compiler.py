import ast_comments as astc

code_cache = {}

def codegen(backend_name: str, loop_source, loop_name, val_map, options):
    cache_key = (backend_name, loop_source)
    if cache_key in code_cache:            
        return code_cache[cache_key]
    
    tree = astc.parse(loop_source)
    # Do frontend trasformations
    from .frontend import sanity_check
    from .frontend import rewrite_prange
    from .frontend import rewrite_range

    sanity_check.visit(tree)
    tree = rewrite_prange.transform(tree)
    tree = rewrite_range.transform(tree)

    # astc.fix_missing_locations(tree)
    # print("New code:", astc.unparse(tree))
    # exit(0)

    f = None
    if backend_name == "triton":
        from .backends.triton.codegen import codegen            
    elif backend_name == "ptx":
        from .backends.ptx.codegen import codegen
    elif backend_name == "cuda":
        from .backends.cuda.codegen import codegen
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
    
    f = codegen(tree, loop_name, val_map, options)
    code_cache[cache_key] = f        
    return f
        

def exec(f, val_map):
    args = list(val_map.values())
    f(*args)