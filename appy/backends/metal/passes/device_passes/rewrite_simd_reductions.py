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
    For each write-back (target = reduction_var):
      - Rewrite to: target = __threadgroup_var[0]  (broadcast result to all threads)
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

        elif (isinstance(child, ast.Assign)
                and isinstance(child.value, ast.Name)
                and child.value.id in reduction_vars):
            # Broadcast: rewrite target = var -> target = __threadgroup_var[0]
            tg_var, _ = reduction_vars[child.value.id]
            new_body.append(ast.Assign(
                targets=child.targets,
                value=ast.Subscript(
                    value=ast.Name(id=tg_var, ctx=ast.Load()),
                    slice=ast.Constant(value=0),
                    ctx=ast.Load()
                ),
                lineno=getattr(child, 'lineno', None)
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
