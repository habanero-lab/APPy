import importlib.util
import tempfile
import sys
from pathlib import Path

def load_module_from_str(code_str: str, module_name: str = "dynamic_module"):
    """
    Dynamically loads a Python module from a source code string.

    Parameters
    ----------
    code_str : str
        The full source code of the module as a string.
    module_name : str, optional
        The name to assign to the module (default: "dynamic_module").

    Returns
    -------
    types.ModuleType
        The imported module object. You can access functions, classes, etc.
        Example: mod.kernel_appy(a, b, c)
    """
    # Create a temporary file to hold the module code
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f"{module_name}.py"
    temp_path.write_text(code_str)

    # Load the module spec and execute
    spec = importlib.util.spec_from_file_location(module_name, temp_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module

def load_func_from_str(code_str: str, func: str):
    m = load_module_from_str(code_str)
    return getattr(m, func)