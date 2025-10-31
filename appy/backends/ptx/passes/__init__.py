def block_loop(tree):
    '''
    Transform loops in the AST to use block-level parallelism if `parallel=True`
    and thread-level parallelism if `simd=True`.
    For example, replace prange loops with assignments to __ctaid_x. If the outer 
    loop is blocked and the inner loop is simd, the inner loop index would be 
    assigned to __tid_x.


    Parameters
    ----------
    tree : ast.AST
        The AST of the Python code to transform.

    Returns
    -------
    ast.AST
        The transformed AST.
    '''
    from .block_loop import BlockLoop
    return BlockLoop().visit(tree)

def attach_types(tree, val_map):
    '''
    Attach type annotations to nodes in the AST. After annotation, 
    type information can be accessed via the `appy_type` attribute of AST nodes.

    Parameters
    ----------
    tree : ast.AST
        The AST of the Python code to transform.

    Returns
    -------
    (ast.AST, dict)
        The transformed AST and a mapping from variable names to their types.
    '''
    from .attach_types import AttachTypes
    visitor = AttachTypes(val_map)
    visitor.visit(tree)
    return tree, visitor.type_map

def to_pseudo_ptx(tree, val_map, type_map):
    '''
    Transform the AST to a pseudo-PTX representation. This includes loading
    kernel parameters into registers and replacing variable references with
    register references.

    Parameters
    ----------
    tree : ast.AST
        The AST of the Python code to transform.
    val_map : dict
        A mapping from variable names to their values.
    type_map : dict
        A mapping from variable names to their types.

    Returns
    -------
    ast.AST
        The transformed AST.
    '''
    from .to_pseudo_ptx import transform
    return transform(tree, val_map, type_map)

def remove_appy(tree):
    '''
    Replaces `appy.prange` calls with `prange` calls in the AST.

    Parameters
    ----------
    tree : ast.AST
        The AST of the Python code to transform.

    Returns
    -------
    ast.AST
        The transformed AST.
    '''
    from .remove_appy import RemoveAPPy
    return RemoveAPPy().visit(tree)

def to_unit_stmts_form(tree):
    '''
    Transform the AST to unit statements form, where each statement contains
    at most one arithmetic operation, function call, logical operation,
    comparison, bitwise operation, or array load/store.

    Parameters
    ----------
    tree : ast.AST
        The AST of the Python code to transform.

    Returns
    -------
    ast.AST
        The transformed AST.
    '''
    from .to_unit_stmts_form import ToUnitStmtsForm
    transformer = ToUnitStmtsForm()
    return transformer.visit(tree)

def add_builtin_imports(tree):
    '''
    Add necessary built-in imports to the AST.

    Parameters
    ----------
    tree : ast.AST
        The AST of the Python code to transform.

    Returns
    -------
    ast.AST
        The transformed AST.
    '''
    from .add_builtin_imports import AddBuiltinImports
    transformer = AddBuiltinImports()
    return transformer.visit(tree)

def codegen_pycuda_imports(tree):
    '''
    Add PyCUDA import statements to the AST.

    Parameters
    ----------
    tree : ast.AST
        The AST of the Python code to transform.

    Returns
    -------
    ast.AST
        The transformed AST.
    '''
    from .codegen_pycuda_imports import AddPyCUDAImports
    transformer = AddPyCUDAImports()
    return transformer.visit(tree)

def codegen_data_movement(tree, val_map):
    '''
    Insert data movement operations (e.g., host to device and device to host
    transfers) into the AST where necessary.

    Parameters
    ----------
    tree : ast.AST
        The AST of the Python code to transform.
    val_map : dict
        A mapping from variable names to their values.

    Returns
    -------
    ast.AST
        The transformed AST.
    '''
    from .codegen_data_movement import InsertDataMovement
    transformer = InsertDataMovement(val_map)
    return transformer.visit(tree)

def codegen_load_kernel(tree, path):
    '''
    Add code to load the kernel from PTX file into the AST.

    Parameters
    ----------   
    tree : ast.AST
        The AST of the Python code to transform.
    path : str
        The path to the PTX file.

    Returns
    -------
    ast.AST
        The transformed AST.
    '''
    from .codegen_load_kernel import AddCodeLoadKernel
    transformer = AddCodeLoadKernel(path)
    return transformer.visit(tree)

def codegen_kernel_launch(tree):
    '''
    Add code to launch the kernel in the AST.

    Parameters
    ----------
    tree : ast.AST
        The AST of the Python code to transform.

    Returns
    -------
    ast.AST
        The transformed AST.
    '''
    from .codegen_kernel_launch import CodegenKernelLaunch
    transformer = CodegenKernelLaunch()
    return transformer.visit(tree)

from .ptx_types import PTXType

def map_py_type_to_ptx_type(var_to_type):
    '''
    Map Python types to PTX types based on the provided value map.

    Parameters
    ----------
    val_map : dict
        A dictionary mapping variable names to their Python values.

    Returns
    -------
    dict
        A dictionary mapping variable names to their corresponding PTX types.
    '''
    py_type_to_ptx_type = {
        'int32': PTXType.S32,
        'float32': PTXType.F32,
        'float64': PTXType.F64,
        'int32*': PTXType.U64,
        'float32*': PTXType.U64,
        'float64*': PTXType.U64,
    }

    var_to_ptx_type = {}
    for var, py_type in var_to_type.items():
        if py_type in py_type_to_ptx_type:
            var_to_ptx_type[var] = py_type_to_ptx_type[py_type]
        else:
            raise NotImplementedError(f"PTX type mapping not implemented for Python type: {py_type}")
    return var_to_ptx_type

def codegen_ptx(tree, val_map, type_map):
    '''
    Generate PTX code from the AST.

    Parameters
    ----------
    tree : ast.AST
        The AST of the Python code to transform.
    val_map : dict
        A mapping from variable names to their values.
    type_map : dict
        A mapping from variable names to their types.   
    Returns
    -------
    str
        The generated PTX code as a string.
    '''    
    from .codegen_ptx import CodegenPTX
    var_to_ptx_type = map_py_type_to_ptx_type(type_map)
    generator = CodegenPTX(val_map, var_to_ptx_type)
    generator.visit(tree)
    return generator.get_ptx_code()