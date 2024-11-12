import ast_comments as ast
import inspect
from appy.codegen.triton.gen_code import TritonBackend

libname = 'torch'  # can be either 'torch' or 'cupy'

def compile_from_src(src, **options):
    tree = ast.parse(src)
    args = {}
    backend = TritonBackend(tree, args, **options)
    module = backend.codegen()
    return module

def compile(fn, dump_code=False, verbose=False):
    '''
    Compile an annotated function and returns a new function that executes GPU kernels
    '''
    source_code = inspect.getsource(fn)
    module = compile_from_src(source_code)  # module includes both host and device code
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