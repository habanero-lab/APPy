import os
import re
import sys
import torch
import inspect
import ast_comments as ast
import importlib.util
from pathlib import Path
from slap.codegen.triton import TritonBackend

compiled = {}

def compile(fn, args, dump_code=True, verbose=False):
    if verbose:
        print(f'[jit] Compile function {fn.__name__} with type signature {[type(x) for x in args]}')
    src = inspect.getsource(fn)
    #arg_names = get_arg_names(src)
    #src = constant_prop(src, arg_names, args)
    #print(src)
    tree = ast.parse(src)
    
    backend = TritonBackend(tree, args)
    module = backend.codegen()
    if dump_code:
        print(module)
    #exit(1)
    fn = 'slap_kernel.py'
    Path(fn).write_text(module)
    spec = importlib.util.spec_from_file_location("module.name", fn)
    foo = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    if verbose:
        print("[jit] Done compiling")
    return foo.kernel

def get_arg_names(src):
    defline = ''
    for line in src.split('\n'):
        if line.startswith('@'):
            continue
        if line.startswith('def '):
            defline = line
            break
    result = re.search(r'\((.*)\)', defline)
    args_str = result.groups()[0]
    items = list(map(lambda x: x.strip(), args_str.split(',')))
    return items
    

def constant_prop(src, arg_names, arg_values):
    for i,arg in enumerate(arg_names):
        value = arg_values[i]
        if not isinstance(value, torch.Tensor):
            continue

        for dim in range(len(value.shape)):
            src = src.replace(f'{arg}.shape[{dim}]', str(value.shape[dim]))
    return src
        
def _jit(fn):
    def slap_kernel(*args):
        if fn not in compiled:
            compiled[fn] = compile(fn, args)
        return compiled[fn](*args)
    return slap_kernel

def jit(fn=None, dump_code=False, verbose=False):
    if fn:
        return _jit(fn)
    else:
        #print('return arg version')
        def jit_with_args(fn1):
            def slap_kernel(*args):
                if fn1 not in compiled:
                    compiled[fn1] = compile(fn1, args, dump_code=dump_code, verbose=verbose)
                return compiled[fn1](*args)
            return slap_kernel
        return jit_with_args
