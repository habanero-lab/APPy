# AST imports
import inspect
import textwrap
import ast_comments as ast

# AST passes
from ast_transforms import remove_func_decorator
from .midend import replace_pfor

# Globals
from .__version__ import __version__
_options = None

def _kernel_launch(loop_source, loop_name, scope, global_scope):
    """
    Runtime stub invoked when a prange loop is encountered.

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

    Behavior
    --------
    When appy._options["dry_run"] is True:
        - Simply executes the loop source as regular Python code in the given scope.
    """

    if _options.get("dry_run", False):
        # In dry_run mode, just execute the loop source in the caller's scope
        try:
            code_obj = compile(loop_source, filename=f"<{loop_name}>", mode="exec")        
            exec(code_obj, global_scope or scope, scope)
        except Exception as e:
            raise RuntimeError(f"Error executing loop {loop_name} in dry_run mode: {e}")
    else:
        raise NotImplementedError(
            f"__appy_kernel_launch: non-dry-run mode not yet implemented for {loop_name}"
        )


def compile_loops(fn, **options):
    # 1. Get source and parse into AST
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)

    # 2. Apply transformation
    tree = replace_pfor.transform(tree)
    tree = remove_func_decorator.transform(tree)

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
def prange(*args):
    return range(*args)

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
