# AST imports
import inspect
import textwrap
import ast_comments as ast

# AST passes
import ast_transforms as at
from .frontend import replace_pfor_with_stub
from .backends.base import Backend

# Globals
from .__version__ import __version__
_options = None

def _kernel_launch(loop_source, loop_name, scope, global_scope):
    """
    Runtime stub invoked when a prange loop is encountered. The loop source 
    is compiled and dynamically executed.

    Parameters
    ----------
    loop_source : str
        The original loop source code as a string (e.g., 'for i in prange(...): ...')
    loop_name : str
        Unique name for this loop (e.g., 'kernel_loop_1')
    scope : dict
        The locals() dictionary of the calling function.
    global_scope : dict 
        The global variables of the caller.
    """
    used_names = at.get_used_names(ast.parse(loop_source).body[0])
    merged_scope = global_scope | scope
    val_map = {k: merged_scope[k] for k in used_names if k in merged_scope}

    tree = ast.parse(loop_source)

    backend = Backend.create_backend(_options["backend"])
    target_code_ast = backend.codegen(tree, metadata={
        "loop_name": loop_name,
        "local_scope": scope,
        "global_scope": global_scope,
        "val_map": val_map,
        "options": _options,
    })

    if _options.get("dump_code"):
        print(f"--- Dumped code for loop {loop_name} ---")
        print(ast.unparse(target_code_ast))
        print(f"--- End of dumped code for loop {loop_name} ---")

    if _options.get("dry_run"):
        # In dry_run mode, just execute the loop source in the caller's scope
        try:
            code_obj = compile(loop_source, filename=f"<{loop_name}>", mode="exec")        
            exec(code_obj, global_scope or scope, scope)
        except Exception as e:
            raise RuntimeError(f"Error executing loop {loop_name} in dry_run mode: {e}")
    else:
        #f = load_func_from_str(target_code, "kernel_appy")
        filtered_scope = {k: v for k,v in merged_scope.items() if k != '__name__' and k != '__loader__'}
        backend.exec(target_code_ast, filtered_scope)
        
        # f = ns['kernel_appy']
        
        # args = [merged_scope[x] for x in used_names if x in merged_scope]
        # f(*args)

def compile_loops(fn, **options):
    # 1. Get source and parse into AST
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)

    # 2. Apply transformation
    tree = at.hoist_shape_attr(tree)
    tree = at.remove_func_decorator(tree)
    tree = replace_pfor_with_stub.transform(tree)

    print("new code:", ast.unparse(tree))

    # 3. Compile the new AST into a code object
    code = compile(tree, filename="<ast>", mode="exec")

    # 4. Create a namespace for execution
    namespace = {}
    exec(code, fn.__globals__, namespace)

    # 5. Return the new function object
    new_fn = namespace[fn.__name__]
    return new_fn

def set_default_options(options):
    options.setdefault("backend", "triton")
    options.setdefault("dry_run", False)
    options.setdefault("auto_transfer", True)
    options.setdefault("dump_code", False)

    global _options
    _options = options

def jit(fn=None, **options):
    set_default_options(options)
    if fn:
        return compile_loops(fn)
    else:
        def jit_with_args(fn1):
            return compile_loops(fn1, **options)
        return jit_with_args

# Built-in functions
def vidx(start, stepsize, bound=None):
    if bound:
        r = slice(start, min(bound, start+stepsize))
    else:
        r = slice(start, start+stepsize)
    return r

def prange(*args, simd=False):
    return range(*args)

built_in_range = range

def range(*args, parallel=False, simd=False):
    return built_in_range(*args)

# Data transfer functions
def to_gpu(*args):
    if len(args) == 1:
        return _to_gpu(args[0])
    else:
        return [_to_gpu(a) for a in args]

def _to_gpu(a):
    import torch
    if f"{type(a).__module__}.{type(a).__name__}" == 'numpy.ndarray':
        return torch.from_numpy(a).to('cuda')
    elif type(a) == torch.Tensor:
        return a.to('cuda')
    elif type(a) == list:
        return torch.tensor(a).to('cuda')
    elif type(a) in [int, float]:
        return a
    else:
        assert False, "Unsupported type to transfer to GPU"

def to_cpu(*args):
    if len(args) == 1:
        return _to_cpu(args[0])
    else:
        return [_to_cpu(a) for a in args]

def _to_cpu(a):
    import torch
    if type(a) == torch.Tensor:
        if a.ndim == 0:
            return a.item()
        else:
            return a.to('cpu').numpy()
    else:
        return a

# Dummy functions
def ptx_ld_param_u32(a):
    return a

def ptx_ld_param_u64(a):
    return a