import sys
import inspect
import ast_comments as ast
import importlib
from pathlib import Path
import black
from appy.codegen.triton.gen_code import TritonBackend
from . import config

def compile_from_src(src, **options):
    tree = ast.parse(src)
    args = {}
    backend = TritonBackend(tree, args, **options)
    module = backend.codegen()
    module = black.format_str(module, mode=black.Mode())
    return module

def compile(fn, args, dump_code=False, verbose=False, **options):
    '''
    Compile an annotated function and returns a new function that executes GPU kernels
    '''
    if options.get('lib', 'torch') == 'cupy':
        config.tensorlib = 'cupy'
    
    if options.get('use_file'):
        # Use an existing compiled file
        filename = options.get('use_file')
    else:
        source_code = inspect.getsource(fn)
        module = compile_from_src(source_code, **options)  # module includes both host and device code
        if dump_code:
            print(module)
        filename = f".appy_kernels/{fn.__name__}.py"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_text(module, encoding='utf-8')

    spec = importlib.util.spec_from_file_location("module.name", filename)
    m = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = m
    spec.loader.exec_module(m)

    # Note: stack[1] is `inner`, and stack[2] is the user code caller
    user_globals = inspect.stack()[2].frame.f_globals
    
    # Add the missing globals into the new module
    for k, v in user_globals.items():
        if k not in m.__dict__:
            m.__dict__[k] = v
    
    if verbose:
        print("[jit] Done compiling")
    compiled = getattr(m, fn.__name__)
    return compiled


compiled_funcs = {}

def _jit(fn):
    def inner(*args):
        key = f"{fn}+{get_type_sig(*args)}"
        if key not in compiled_funcs:
            compiled_funcs[key] = compile(fn, args)
        return compiled_funcs[key](*args)

    inner.__name__ = fn.__name__
    return inner

def jit(fn=None, dump_code=None, verbose=None, **options):
    if fn:
        return _jit(fn)
    else:
        # if dump_code != None:
        #     config.configs['dump_code'] = dump_code
        # if verbose != None:
        #     config.configs['verbose'] = verbose

        # print('return arg version')
        def jit_with_args(fn1):
            def inner(*args):
                key = f"{fn1}+{get_type_sig(*args)}"
                if key not in compiled_funcs:
                    compiled_funcs[key] = compile(
                        fn1, args, dump_code=dump_code, verbose=verbose, **options
                    )
                return compiled_funcs[key](*args)

            inner.__name__ = fn1.__name__
            return inner

        return jit_with_args


def get_type_str(v):
    return f"{type(v).__module__}.{type(v).__name__}"

def is_type(v, ty):
    return get_type_str(v) == ty

def get_type_sig(*args):
    sigs = []
    for arg in args:
        if is_type(arg, "torch.Tensor"):
            sigs.append(f"<{arg.dtype}*{arg.dim()}>")
        elif isinstance(arg, int):
            sigs.append(f"{arg}")
        else:
            sigs.append(f"{type(arg)}")
    return ",".join(sigs)

# Special functions
def step(start, stepsize, bound=None):
    if bound:
        r = slice(start, min(bound, start+stepsize))
    else:
        r = slice(start, start+stepsize)
    return r

def debug_barrier():
    pass

def atomic_add(a, offset, b):
    a[offset] += b

vidx = step

# Data transfer functions
def to_gpu(a):
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

def to_cpu(a):
    import torch
    if type(a) == torch.Tensor:
        if a.ndim == 0:
            return a.item()
        else:
            return a.to('cpu').numpy()
    else:
        return a
