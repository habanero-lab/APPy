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