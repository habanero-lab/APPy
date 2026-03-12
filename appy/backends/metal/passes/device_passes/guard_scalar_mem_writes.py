import ast


class GuardScalarMemWrites(ast.NodeTransformer):
    """
    Wrap scalar subscript assignments with `if (lane == 0)` so that only
    one thread per SIMD group performs the write.

    Recurses into non-SIMD for loops and if/else branches, but skips SIMD
    inner loops entirely — writes inside a SIMD loop target distinct
    per-lane elements and must not be guarded.
    """

    def visit_For(self, node):
        if hasattr(node, 'pragma') and node.pragma.get('simd'):
            return node  # per-lane writes inside SIMD loops are safe
        node.body = [self.visit(stmt) for stmt in node.body]
        return node

    def visit_Assign(self, node):
        target = node.targets[0]
        is_user_mem_write = (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and not target.value.id.startswith('__threadgroup_')
        )
        if is_user_mem_write:
            return ast.If(
                test=ast.Compare(
                    left=ast.Name(id='lane', ctx=ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=0)]
                ),
                body=[node],
                orelse=[]
            )
        return node


def transform(loop):
    transformer = GuardScalarMemWrites()
    loop.body = [transformer.visit(stmt) for stmt in loop.body]
    return loop
