import ast
from astpass.passes.shape_analysis import AnalyzeAssignShapes

class AnalyzeAssignShapesWithBuffer(AnalyzeAssignShapes):
    def __init__(self, rt_vals):
        self.rt_vals = rt_vals
        super().__init__(rt_vals)

    def visit_Assign(self, node):
        # Special handling for "xx = appy.buffer(size, dtype)"
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "appy.buffer":
            self.visit_appy_buffer(node.value)
            self.var_shapes[node.targets[0].id] = self.node_shapes[node.value]
        else:
            super().visit_Assign(node)

    def visit_appy_buffer(self, node: ast.Call):
        assert len(node.args) == 2
        size, dtype = node.args
        if isinstance(size, ast.Constant):
            self.node_shapes[node] = (size.value,)
        elif isinstance(size, ast.Name):
            assert size.id in self.rt_vals
            self.node_shapes[node] = self.rt_vals[size.id]
        else:
            raise RuntimeError(f"Unsupported buffer call: {ast.unparse(node)}")
        
def analyze(tree, rt_vals):
    visitor = AnalyzeAssignShapesWithBuffer(rt_vals)
    visitor.visit(tree)
    return visitor.node_shapes