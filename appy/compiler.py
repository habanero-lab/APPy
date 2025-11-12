code_cache = {}

def codegen(backend_name: str, loop_source, loop_name, val_map, options):
    cache_key = (backend_name, loop_source)
    if cache_key in code_cache:            
        return code_cache[cache_key]
    
    # Do frontend trasformations
    # TODO

    f = None
    if backend_name == "triton":
        from .backends.triton.codegen import codegen            
    elif backend_name == "ptx":
        from .backends.ptx.codegen import codegen
    elif backend_name == "cuda":
        from .backends.cuda.codegen import codegen
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
    
    f = codegen(loop_source, loop_name, val_map, options)
    code_cache[cache_key] = f        
    return f
        

def exec(f, val_map):
    args = list(val_map.values())
    f(*args)