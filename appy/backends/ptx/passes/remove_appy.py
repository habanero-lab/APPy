import ast

class RemoveAPPy(ast.NodeTransformer):
    def visit_Call(self, node):
        # Replaces `appy.prange` with `prange`
        if isinstance(node.func, ast.Attribute):
            if node.func.value.id == "appy" and node.func.attr == "prange":
                new_node = ast.Call(
                    func=ast.Name(id="prange", ctx=ast.Load()),
                    args=node.args,
                    keywords=node.keywords
                )
                ast.copy_location(new_node, node)
                return new_node
        return self.generic_visit(node)
    
    