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
    cache_key = (loop_source, types, dtypes, os.environ.get("TRITON_INTERPRET"))
    if cache_key in code_cache:            
        f, code_src = code_cache[cache_key]
    else:   
        code_src = Path(f"{Path(__file__).parent}/sample_kernels/gelu.py").read_text()
        m = load_module_from_str(code_src)
        f = getattr(m, loop_name)

        code_cache[cache_key] = f, code_src

    return f, code_src