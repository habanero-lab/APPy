import ast_comments as ast
import inspect
from pathlib import Path
import subprocess
import importlib
import sys
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
    source_code = inspect.getsource(fn)
    module = compile_from_src(source_code, **options)  # module includes both host and device code
    if dump_code:
        print(module)
    filename = f".appy_kernels/{fn.__name__}.py"
    Path(filename).write_text(module, encoding='utf-8')
    
    #subprocess.run(["black", filename], capture_output=True, text=True)
    subprocess.run(["black", filename], capture_output=True)
    # exit(1)
    spec = importlib.util.spec_from_file_location("module.name", filename)
    foo = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    if verbose:
        print("[jit] Done compiling")
    compiled = getattr(foo, fn.__name__)
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


def get_matmul_configs(BM, BN, BK):
    return [
        {BM: 128, BN: 256, BK: 32, 'num_stages': 3, 'num_warps': 8},
        {BM: 256, BN: 128, BK: 32, 'num_stages': 3, 'num_warps': 8},
        {BM: 256, BN: 64, BK: 32, 'num_stages': 4, 'num_warps': 8},
        {BM: 64, BN: 256, BK: 32, 'num_stages': 4, 'num_warps': 8},
        {BM: 128, BN: 128, BK: 32, 'num_stages': 4, 'num_warps': 8},


        {BM: 256, BN: 64, BK: 32, 'num_stages': 4, 'num_warps': 4},
        {BM: 64, BN: 256, BK: 32, 'num_stages': 4, 'num_warps': 4},
        {BM: 128, BN: 128, BK: 32, 'num_stages': 4, 'num_warps': 4},
        {BM: 128, BN: 64, BK: 32, 'num_stages': 4, 'num_warps': 4},
        {BM: 64, BN: 128, BK: 32, 'num_stages': 4, 'num_warps': 4},
        {BM: 128, BN: 32, BK: 32, 'num_stages': 4, 'num_warps': 4},
        {BM: 64, BN: 32, BK: 32, 'num_stages': 5, 'num_warps': 2},
    ]