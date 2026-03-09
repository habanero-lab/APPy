import ast
from ..constants import SIMD_WIDTH


def _call_stmt(name, *args):
    """Create a statement: name(arg1, arg2, ...)"""
    return ast.Expr(value=ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()),
        args=[ast.Constant(value=a) for a in args],
        keywords=[]
    ))


def transform(loop):
    """
    Rewrite simd reduction loops in the body of a parallel for loop.

    For each simd reduction inner loop:
      - Keep the strided accumulation loop
      - After it: emit __threadgroup_reduce(var, tg_var, op)
        The unparser expands this to: store lane partial, barrier, tree reduce,
        then propagate final result back into var for all threads.
    Prepend __metal_shared_mem_decl for each reduction variable found.
    """
    new_body = []
    reduction_vars = {}  # var_name -> (tg_var_name, op)

    for child in loop.body:
        if (isinstance(child, ast.For)
                and hasattr(child, 'pragma')
                and child.pragma.get('reduction')):
            var = child.pragma['reduction_var']
            op = child.pragma['reduction']
            tg_var = f'__threadgroup_{var}'
            reduction_vars[var] = (tg_var, op)

            new_body.append(child)
            new_body.append(_call_stmt('__threadgroup_reduce', var, tg_var, op))
        else:
            new_body.append(child)

    # Prepend shared mem declarations for all reduction vars found
    decls = [
        _call_stmt('__metal_shared_mem_decl', tg_var, 'float', SIMD_WIDTH)
        for _, (tg_var, _) in reduction_vars.items()
    ]
    loop.body = decls + new_body
    return loop
