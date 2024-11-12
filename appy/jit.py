import os
import re
import sys
import subprocess
import torch
import inspect
import ast_comments as ast
import importlib.util
from pathlib import Path
from appy.codegen.triton.gen_code import TritonBackend

# import appy.config as config

compiled = {}


def compile(fn, args, dump_code=0, verbose=False, **options):
    # print('options:', options)
    if options.get("use_compiled_file"):
        filename = options.get("use_compiled_file")
        print("use compiled file:", filename)
    else:
        appy_kernel_dir = "./.appy_kernels"
        os.makedirs(appy_kernel_dir, exist_ok=True)
        if verbose:
            print(
                f"[jit] Compile function {fn.__name__} with type signature {[type(x) for x in args]}"
            )
        src = inspect.getsource(fn)
        # src = constant_prop(src, get_arg_names(src), args)
        tree = ast.parse(src)

        backend = TritonBackend(tree, args, **options)
        module = backend.codegen()
        if dump_code:
            print(module)
        filename = f"{appy_kernel_dir}/{fn.__name__}.py"
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


def preprocess(src):
    lines = src.split("\n")
    newsrc = ""
    i = 0
    while i < len(lines):
        line = lines[i]

        m = re.match(r" *\#pragma", line)
        if m:
            nextline = lines[i + 1]
            assert re.match(r" *for ", nextline)
            newsrc += nextline + line + "\n"
            i += 1
        else:
            newsrc += line + "\n"
        i += 1
    return newsrc


def get_arg_names(src):
    defline = ""
    for line in src.split("\n"):
        if line.startswith("@"):
            continue
        if line.startswith("def "):
            defline = line
            break
    result = re.search(r"\((.*)\)", defline)
    args_str = result.groups()[0]
    items = list(map(lambda x: x.strip(), args_str.split(",")))
    items = list(filter(lambda x: "=" not in x, items))

    return items


def constant_prop(src, arg_names, arg_values):
    for i, arg in enumerate(arg_names):
        value = arg_values[i]
        if not isinstance(value, torch.Tensor):
            continue

        src = src.replace(f"{arg}.dtype", str(value.dtype))

        for dim in range(len(value.shape)):
            src = src.replace(f"{arg}.shape[{dim}]", f"{arg}_shape_{dim}")
    return src


def get_type_sig(*args):
    sigs = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            sigs.append(f"<{arg.dtype}*{arg.dim()}>")
        elif isinstance(arg, int):
            sigs.append(f"{arg}")
        else:
            sigs.append(f"{type(arg)}")
    return ",".join(sigs)


