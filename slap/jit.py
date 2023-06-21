import os
import re
import sys
import subprocess
import torch
import inspect
import ast_comments as ast
import importlib.util
from pathlib import Path
from slap.codegen.triton import TritonBackend
import slap.config as config

compiled = {}

def compile(fn, args, dump_code=0, verbose=False):
    if verbose:
        print(f'[jit] Compile function {fn.__name__} with type signature {[type(x) for x in args]}')
    src = inspect.getsource(fn)
    src = preprocess(src)
    src = constant_prop(src, get_arg_names(src), args)
    tree = ast.parse(src)
    
    backend = TritonBackend(tree, args)
    module = backend.codegen()
    if dump_code:
        print(module)
    fn = 'slap_kernel.py'
    Path(fn).write_text(module)
    subprocess.run(['black', fn], capture_output=True, text=True)
    
    spec = importlib.util.spec_from_file_location("module.name", fn)
    foo = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    if verbose:
        print("[jit] Done compiling")
    compiled = foo.kernel
    
    return compiled

def preprocess(src):
    lines = src.split('\n')
    newsrc = ''
    i = 0
    while i < len(lines):
        line = lines[i]
        
        m = re.match(r' *\#pragma', line)
        if m:
            nextline = lines[i+1]
            assert re.match(r' *for ', nextline)
            newsrc += nextline + line + '\n'
            i += 1
        else:
            newsrc += line + '\n'
        i += 1
    return newsrc

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
    items = list(filter(lambda x: '=' not in x, items))
    
    return items
    

def constant_prop(src, arg_names, arg_values):
    for i,arg in enumerate(arg_names):
        value = arg_values[i]
        if not isinstance(value, torch.Tensor):
            continue

        src = src.replace(f'{arg}.dtype', str(value.dtype))

        # for dim in range(len(value.shape)):
        #     src = src.replace(f'{arg}.shape[{dim}]', str(value.shape[dim]))
    return src

def get_type_sig(*args):
    sigs = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            sigs.append(f'<{arg.dtype}*{arg.dim()}>')
        else:
            sigs.append(f'{type(arg)}')
    return ','.join(sigs)
        
def _jit(fn):
    def inner(*args):
        key = f'{fn}+{get_type_sig(*args)}'
        
        if key not in compiled:
            compiled[key] = compile(fn, args)
        return compiled[key](*args)
    inner.__name__ = fn.__name__  
    return inner

def jit(fn=None, dump_code=None, verbose=None):
    if fn:
        return _jit(fn)
    else:
        # if dump_code != None:
        #     config.configs['dump_code'] = dump_code
        # if verbose != None:
        #     config.configs['verbose'] = verbose
            
        #print('return arg version')
        def jit_with_args(fn1):
            def inner(*args):
                key = f'{fn1}+{get_type_sig(*args)}'
                if key not in compiled:
                    compiled[key] = compile(fn1, args, dump_code=dump_code, verbose=verbose)
                return compiled[key](*args)
            inner.__name__ = fn1.__name__  
            return inner
        return jit_with_args
