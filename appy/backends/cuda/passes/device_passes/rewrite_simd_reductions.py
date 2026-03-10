import ast
from ..constants import BLOCK_SIZE


def _call_stmt(name, *args):
    return ast.Expr(value=ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()),
        args=[ast.Constant(value=a) for a in args],
        keywords=[]
    ))


def transform(loop):
    '''
    Rewrites SIMD reduction loops in the body of a parallel for loop.

    For each inner simd reduction loop:
      - Keep the strided accumulation loop
      - After it: emit __cuda_shared_reduce(var, sh_var, op)
        The unparser expands this to a shared-memory tree reduction using
        __syncthreads().
    Prepend __cuda_shared_mem_decl for each reduction variable found.
    '''
    new_body = []
    reduction_vars = {}

    for child in loop.body:
        if (isinstance(child, ast.For)
                and hasattr(child, 'pragma')
                and child.pragma.get('reduction')):
            var = child.pragma['reduction_var']
            op = child.pragma['reduction']
            sh_var = f'__shared_{var}'
            reduction_vars[var] = (sh_var, op)
            new_body.append(child)
            new_body.append(_call_stmt('__cuda_shared_reduce', var, sh_var, op))
        else:
            new_body.append(child)

    decls = [
        _call_stmt('__cuda_shared_mem_decl', sh_var, 'float', BLOCK_SIZE)
        for _, (sh_var, _) in reduction_vars.items()
    ]
    loop.body = decls + new_body
    return loop
