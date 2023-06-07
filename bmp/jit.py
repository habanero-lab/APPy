import os
import re
import sys
import torch
import inspect
import textwrap
import ast_comments as ast
import importlib.util
from bmp.codegen.cuda import CUDABackend

compiled = {}

def compile(fn, args):
    print(f'[jit] Compile function {fn.__name__} with type signature {args}')
    src = inspect.getsource(fn)
    arg_names = get_arg_names(src)
    src = constant_prop(src, arg_names, args)
    print(src)
    tree = ast.parse(src)
    
    print(ast.dump(tree))
    backend = CUDABackend(tree)
    backend.codegen()

def get_arg_names(src):
    lines = src.split('\n')
    assert 'def ' in lines[1]
    result = re.search(r'\((.*)\)', lines[1])
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
        

def jit(fn):
    def inner(*args):
        if fn not in compiled:
            compiled[fn] = compile(fn, args)
        return compiled[fn](*args)
    
    return inner
    
