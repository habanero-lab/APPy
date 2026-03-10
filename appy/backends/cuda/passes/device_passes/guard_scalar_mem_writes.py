import ast


def transform(loop):
    '''
    In SIMD mode, each parallel iteration is handled by BLOCK_SIZE threads.
    Direct memory writes in the loop body (outside a simd inner loop) would be
    executed by all threads, causing a race condition.  Wrap them with
    `if (lane == 0)` so only thread 0 of each block performs the write.
    '''
    new_body = []
    for child in loop.body:
        target = child.targets[0] if isinstance(child, ast.Assign) else None
        is_user_mem_write = (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and not target.value.id.startswith('__shared_')
        )
        if is_user_mem_write:
            guard = ast.If(
                test=ast.Compare(
                    left=ast.Name(id='lane', ctx=ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=0)]
                ),
                body=[child],
                orelse=[]
            )
            new_body.append(guard)
        else:
            new_body.append(child)
    loop.body = new_body
    return loop
