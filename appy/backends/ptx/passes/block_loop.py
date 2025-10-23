import ast
import ast_transforms as at

class BlockLoop(ast.NodeTransformer):
    def __init__(self):
        self.name_map = {}

    def get_loop_pragma(self, node):
        return {"parallel": True}
    
    def visit_For(self, node):
        pragma = self.get_loop_pragma(node)
        if not pragma.get("simd", False):
            # Rewrite the loop index to __ctaid_x and insert 
            # necessary assignments in the beginning of the loop body.
            # Note: for nested prange loops, we'd use __ctaid_y, __ctaid_z, etc.
            # But that is not handled here for simplicity.            
            self.name_map[node.target.id] = "__ctaid_x"
            assign = ast.Assign(
                targets=[ast.Name(id=node.target.id, ctx=ast.Store())], 
                value=ast.Name(id="__ctaid_x", ctx=ast.Load())
            )
            node.body.insert(0, assign)            
            node.target = ast.Name(id="__ctaid_x", ctx=ast.Store())            
        else:
            raise NotImplementedError("SIMD loops not implemented in BlockLoop pass.")
        
        self.generic_visit(node)
        ast.fix_missing_locations(node)
        return node
    
    def visit_Name(self, node):
        if node.id in self.name_map:
            new_node = ast.Name(id=self.name_map[node.id], ctx=node.ctx)
            ast.copy_location(new_node, node)
            return new_node
        return node

