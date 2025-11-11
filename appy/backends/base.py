class Backend():
    code_cache = {}
        
    @staticmethod
    def codegen(backend_name: str, loop_source, loop_name, val_map, options):
        cache_key = (backend_name, loop_source)
        if cache_key in Backend.code_cache:            
            return Backend.code_cache[cache_key]
        
        f = None
        if backend_name == "triton":
            from . import triton
            f = triton.codegen(loop_source, loop_name, val_map, options)
        elif backend_name == "ptx":
            from . import ptx
            f = ptx.codegen(loop_source, loop_name, val_map, options)
        elif backend_name == "cuda":
            from . import cuda
            f = cuda.codegen(loop_source, loop_name, val_map, options)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
        Backend.code_cache[cache_key] = f        
        return f
        
    @staticmethod
    def exec(f, val_map):
        args = list(val_map.values())
        f(*args)
