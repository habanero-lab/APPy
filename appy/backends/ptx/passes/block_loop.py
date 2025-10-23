import ast

class BlockLoop(ast.NodeTransformer):
    def get_loop_pragma(self, node):
        return {"parallel": True}
    
    def visit_For(self, node):
        pragma = self.get_loop_pragma(node)
        if not pragma.get("simd", False):
            # Rewrite the loop index to __ctaid_x and insert 
            # necessary assignments in the beginning of the loop body.
            # Note: for nested prange loops, we'd use __ctaid_y, __ctaid_z, etc.
            # But that is not handled here for simplicity.
            simt_index = ast.Name(id="__ctaid_x", ctx=ast.Load())
            assign = ast.Assign(targets=[node.target], value=simt_index)
            node.body.insert(0, assign)
            node.target = simt_index
        self.generic_visit(node)
        ast.fix_missing_locations(node)
        return node

