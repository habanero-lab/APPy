import os
import sys
import shutil
import platform
import ast_comments as astc
import ast_transforms as at
from .utils import load_module_from_str, pretty_dump

def codegen(backend_name: str, loop_source, loop_name, local_scope, global_scope, options):
    merged_scope = global_scope | local_scope
    used_names = at.get_used_names(astc.parse(loop_source))
    val_map = {k: merged_scope[k] for k in used_names if k in merged_scope}

    if sys.platform == "darwin":
        if platform.machine() != "arm64":
            raise RuntimeError("macOS with x86_64 is not supported")
        
        from .backends.metal.codegen import codegen as metal_codegen
        f, code_src = metal_codegen(loop_source, loop_name, val_map, options)
        
    elif sys.platform.startswith("linux"):
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("NVIDIA GPU not found")
        
        from .backends.triton.codegen import codegen as triton_codegen
        f, code_src = triton_codegen(loop_source, loop_name, val_map, options)


    if options.get("dry_run"):
        # In dry_run mode, just execute the loop source in the caller's scope
        try: 
            code_obj = compile(loop_source, filename=f"<{loop_name}>", mode="exec")             
            exec(code_obj, local_scope, global_scope)
        except Exception as e:
            raise RuntimeError(f"Error executing loop {loop_name} in dry_run mode: {e}")        
        return

    if sys.platform == "darwin":
        from .backends.metal.codegen import exec as metal_exec
        metal_exec(f, val_map)
    else:
        args = list(val_map.values())
        f(*args)