import ast
from ..constants import SIMD_WIDTH


def _call_stmt(name, *args):
    """Create a statement: name(arg1, arg2, ...)"""
    return ast.Expr(value=ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()),
        args=[ast.Constant(value=a) for a in args],
        keywords=[]
    ))


def _subscript_assign(arr_name, idx_name, value_node):
    """Create: arr_name[idx_name] = value_node"""
    return ast.Assign(
        targets=[ast.Subscript(
            value=ast.Name(id=arr_name, ctx=ast.Load()),
            slice=ast.Name(id=idx_name, ctx=ast.Load()),
            ctx=ast.Store()
        )],
        value=value_node,
        lineno=None
    )


def transform(loop):
    """
    Rewrite simd reduction loops in the body of a parallel for loop.

    For each simd reduction inner loop:
      - Keep the strided accumulation loop
      - After it: store lane partial to threadgroup mem, barrier, tree reduce
      - Then propagate final result back: var = __threadgroup_var[0]
        (all threads now hold the fully reduced value for subsequent use)
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
            new_body.append(_subscript_assign(
                tg_var, 'lane', ast.Name(id=var, ctx=ast.Load())))
            new_body.append(_call_stmt('__metal_threadgroup_barrier'))
            new_body.append(_call_stmt('__metal_tree_reduce', tg_var, op))
            # Propagate reduced value back to local var so all threads see the
            # final result; subsequent write-backs (e.g. y[i] = s) work as-is.
            metal_type = next(
                (stmt.targets[0].metal_type
                 for stmt in new_body
                 if isinstance(stmt, ast.Assign)
                 and isinstance(stmt.targets[0], ast.Name)
                 and stmt.targets[0].id == var
                 and hasattr(stmt.targets[0], 'metal_type')),
                'float'
            )
            target_name = ast.Name(id=var, ctx=ast.Store())
            target_name.metal_type = metal_type
            new_body.append(ast.Assign(
                targets=[target_name],
                value=ast.Subscript(
                    value=ast.Name(id=tg_var, ctx=ast.Load()),
                    slice=ast.Constant(value=0),
                    ctx=ast.Load()
                ),
                lineno=None
            ))
        else:
            new_body.append(child)

    # Prepend shared mem declarations for all reduction vars found
    decls = [
        _call_stmt('__metal_shared_mem_decl', tg_var, 'float', SIMD_WIDTH)
        for _, (tg_var, _) in reduction_vars.items()
    ]
    loop.body = decls + new_body
    return loop
