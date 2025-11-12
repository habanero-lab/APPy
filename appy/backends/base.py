class Backend():
    code_cache = {}
        
    @staticmethod
    def codegen(backend_name: str, loop_source, loop_name, val_map, options):
        cache_key = (backend_name, loop_source)
        if cache_key in Backend.code_cache:            
            return Backend.code_cache[cache_key]
        
        f = None
        if backend_name == "triton":
            from .triton.codegen import codegen            
        elif backend_name == "ptx":
            from .ptx.codegen import codegen
        elif backend_name == "cuda":
            from .cuda.codegen import codegen
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        f = codegen(loop_source, loop_name, val_map, options)
        Backend.code_cache[cache_key] = f        
        return f
        
    @staticmethod
    def exec(f, val_map):
        args = list(val_map.values())
        f(*args)
