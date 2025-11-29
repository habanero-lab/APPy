# import importlib.util
# import tempfile
# import sys
# from pathlib import Path

# def load_module_from_str(code_str: str, module_name: str = "dynamic_module"):
#     """
#     Dynamically loads a Python module from a source code string.

#     Parameters
#     ----------
#     code_str : str
#         The full source code of the module as a string.
#     module_name : str, optional
#         The name to assign to the module (default: "dynamic_module").

#     Returns
#     -------
#     types.ModuleType
#         The imported module object. You can access functions, classes, etc.
#         Example: mod.kernel_appy(a, b, c)
#     """
#     # Create a temporary file to hold the module code
#     temp_dir = Path(tempfile.gettempdir())
#     temp_path = temp_dir / f"{module_name}.py"
#     temp_path.write_text(code_str)

#     # Load the module spec and execute
#     spec = importlib.util.spec_from_file_location(module_name, temp_path)
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[module_name] = module
#     spec.loader.exec_module(module)

#     return module

# def load_func_from_str(code_str: str, func: str):
#     m = load_module_from_str(code_str)
#     return getattr(m, func)


import importlib.util
import tempfile
import os, sys
import uuid

def load_module_from_str(code_str, namespace=None):
    """
    Dynamically load a Python module from a code string.
    
    Args:
        code_str (str): The Python source code defining the module.
        namespace (dict, optional): Variables to inject into the module's global scope.
    
    Returns:
        module: The dynamically loaded Python module object.
    """
    # Create a unique temporary file
    temp_dir = tempfile.gettempdir()
    module_name = f"appy_temp_{uuid.uuid4().hex}"
    temp_path = os.path.join(temp_dir, f"{module_name}.py")

    # Write the code to the temporary file
    with open(temp_path, "w") as f:
        f.write(code_str)

    # Dynamically import the file
    spec = importlib.util.spec_from_file_location(module_name, temp_path)
    module = importlib.util.module_from_spec(spec)

    # Inject namespace if provided
    if namespace:
        module.__dict__.update(namespace)

    # Execute the module code
    spec.loader.exec_module(module)

    # Optionally, clean up file (optional — leave if you want caching)
    # os.remove(temp_path)

    return module

def pretty_dump(src, loop_name):
    print(f"--- Dumped code for loop {loop_name} ---")
    # Try to use black to format if installed
    try:
        import black
        print(black.format_str(src, mode=black.FileMode()))
    except ImportError:
        print(src)
    print(f"--- End of dumped code for loop {loop_name} ---")


import re
from collections import OrderedDict

# Code by ChatGPT :)
def parse_pragma(pragma_str):
    # Remove "#pragma" and any trailing colon
    pragma_str = pragma_str.strip().lstrip("#pragma").strip()
    
    tokens = []
    i = 0
    while i < len(pragma_str):
        if pragma_str[i].isspace():
            i += 1
            continue
        # Match a clause with parentheses like to(a,b)
        if match := re.match(r'(\w+)\s*\(([^)]*)\)', pragma_str[i:]):
            key, val = match.group(1), match.group(2)
            val_tuple = tuple(v.strip() for v in val.split(',') if v.strip())
            
            if key in ['shared', 'to', 'from']:
                tokens.append((key, list(val_tuple)))
            else:
                tokens.append((key, val_tuple if len(val_tuple) > 1 else val_tuple[0]))
                
            # if len(val_tuple) == 0:
            #     tokens.append((key, val_tuple))
            # else:
            #     tokens.append((key, val_tuple if len(val_tuple) > 1 else val_tuple[0]))
            i += match.end()
        else:
            # Match single-word tokens like 'parallel' or 'simd'
            match = re.match(r'\w+', pragma_str[i:])
            if match:
                tokens.append((match.group(0), True))
                i += match.end()
            else:
                raise ValueError(f"Unexpected syntax at: {pragma_str[i:]}")
    
    # Postprocess to merge 'parallel for' into 'parallel_for'
    result = OrderedDict()
    skip_next = False
    for idx, (key, val) in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if key == 'parallel' and idx + 1 < len(tokens) and tokens[idx + 1][0] == 'for':
            result['parallel_for'] = True
            skip_next = True
        else:
            result[key] = val

    # Check for unrecognized clauses
    recognized_clauses = {'parallel_for', 'simd', 'block', 'reduction', 'shared', 'to', 'from', 'atomic'}
    for key in result:
        if key not in recognized_clauses:
            raise ValueError(f"Unrecognized pragma clause: `{key}` in `{pragma_str}`")
        
    return result

# Code by ChatGPT :)
def dict_to_pragma(pragma_dict):
    clauses = []

    # Handle 'parallel_for' specially, converting to 'parallel for'
    if pragma_dict.get('parallel_for'):
        clauses.extend(['parallel', 'for'])

    for key, val in pragma_dict.items():
        if key == 'parallel_for':
            continue  # already handled

        if val is True:
            clauses.append(key)
        elif isinstance(val, tuple):
            val_str = ', '.join(val)
            clauses.append(f'{key}({val_str})')
        else:
            clauses.append(f'{key}({val})')

    return '#pragma ' + ' '.join(clauses)