import os
import ast_comments as astc
from .utils import load_module_from_str, pretty_dump

code_cache = {}

def codegen(backend_name: str, loop_source, loop_name, val_map, options):
    types = tuple([type(v) for v in val_map.values()])
    backend_name_key = backend_name
    if backend_name == "triton" and os.environ.get("TRITON_INTERPRET") == "1":
        # If TRITON_INTERPRET is set to 1, add it to the key as well
        backend_name_key = f"{backend_name}_interpret"

    cache_key = (backend_name_key, loop_source, types)
    if cache_key in code_cache:            
        f, code_src = code_cache[cache_key]
    else:   
        tree = astc.parse(loop_source)
        # Do frontend trasformations
        from .frontend import sanity_check
        #from .frontend import rewrite_prange
        from .frontend import rewrite_range
        from .frontend import rewrite_aug_assign
        from .frontend import attach_pragma

        sanity_check.visit(tree)
        tree = rewrite_aug_assign.transform(tree)
        #tree = rewrite_prange.transform(tree)
        tree = rewrite_range.transform(tree)
        attach_pragma.visit(tree)

        # astc.fix_missing_locations(tree)
        # print("New code:", astc.unparse(tree))
        # exit(0)

        if backend_name == "triton":
            from .backends.triton.codegen import codegen            
        elif backend_name == "ptx":
            from .backends.ptx.codegen import codegen
        elif backend_name == "cuda":
            from .backends.cuda.codegen import codegen
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        code_src = codegen(tree, loop_name, val_map, options)
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

    if options.get("dry_run"):
        # In dry_run mode, just execute the loop source in the caller's scope
        try:
            import importlib
            code_obj = compile(loop_source, filename=f"<{loop_name}>", mode="exec") 
            val_map["appy"] = importlib.import_module("appy")  
            exec(code_obj, val_map)
        except Exception as e:
            raise RuntimeError(f"Error executing loop {loop_name} in dry_run mode: {e}")
        return

    args = list(val_map.values())
    f(*args)        