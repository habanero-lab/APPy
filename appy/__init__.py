# AST imports
import inspect
import textwrap
import ast
import ast_comments as astc

# AST passes
import ast_transforms as at
from .frontend import replace_pfor_with_stub
from .frontend import hoist_shape_attr

# Globals
from .__version__ import __version__
from . import dispatcher


def _kernel_launch(loop_source, loop_name, scope, global_scope, options):
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
    used_names = at.get_used_names(ast.parse(loop_source))
    merged_scope = global_scope | scope
    val_map = {k: merged_scope[k] for k in used_names if k in merged_scope}
    dispatcher.codegen(options.get("backend"), loop_source, loop_name, val_map, options)
        
        # f = ns['kernel_appy']
        
        # args = [merged_scope[x] for x in used_names if x in merged_scope]
        # f(*args)

def rewrite_loops(fn, **options):
    # 1. Get source and parse into AST
    source = textwrap.dedent(inspect.getsource(fn))
    tree = astc.parse(source)

    # 2. Apply transformation
    tree = hoist_shape_attr.transform(tree)
    tree = at.remove_func_decorator(tree)
    tree = replace_pfor_with_stub.transform(tree, options)

    newcode = ast.unparse(tree)
    print("new code:", newcode)
    # 3. Compile the new code into a code object
    code = compile(newcode, filename="<string>", mode="exec")

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
    options.setdefault("dump_code_to_file", None)

def jit(fn=None, **options):
    set_default_options(options)
    if fn:
        return rewrite_loops(fn, **options)
    else:
        def jit_with_args(fn1):
            return rewrite_loops(fn1, **options)
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