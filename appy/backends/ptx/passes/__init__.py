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
    from .to_pseudo_ptx import ToPseudoPTX
    return ToPseudoPTX(val_map, type_map).visit(tree)