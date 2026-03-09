import ast


def transform(loop):
    """
    Wrap top-level subscript assignments in a parallel for body with
    `if (lane == 0)` so that only one thread per row performs the write.

    This is needed in SIMD mode where each parallel iteration is handled
    by SIMD_WIDTH threads. Any direct memory write in the body (outside a
    simd inner loop) would otherwise be executed by all threads, causing
    a race condition.
    """
    new_body = []
    for child in loop.body:
        target = child.targets[0] if isinstance(child, ast.Assign) else None
        is_user_mem_write = (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and not target.value.id.startswith('__threadgroup_')
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
