import ast
from . import metal_math

class AttachTypes(ast.NodeVisitor):
    def __init__(self, val_map):
        self.val_map = val_map
        self.name_to_type = {}
        self.verbose = True

    def visit_For(self, node):
        # loop index variable is int64 type by default
        self.name_to_type[node.target.id] = 'int32'
        for child in node.body:
            self.visit(child)
        return node

    def visit_Call(self, node):
        func = None
        if isinstance(node.func, ast.Attribute):
            func = node.func.attr
        elif isinstance(node.func, ast.Name):
            func = node.func.id

        assert hasattr(metal_math, func), "Unknown function: " + func
        arg_count = getattr(metal_math, func).__code__.co_argcount
        assert len(node.args) == arg_count, f"Function {func} takes {arg_count} arguments, but got {len(node.args)}"

        for arg in node.args:
            self.visit(arg)
        # Assume the return value has the same type as the first argument
        node.appy_type = node.args[0].appy_type

    def visit_Constant(self, node):
        val = node.value
        if isinstance(val, int):
            node.appy_type = 'int32'
        elif isinstance(val, float):
            node.appy_type = 'float32'
        else:
            assert False
    
    def visit_Subscript(self, node):
        assert isinstance(node.value, ast.Name) and node.value.id in self.val_map
        dtype = self.val_map[node.value.id].dtype
        if str(dtype) == 'float32':
            node.appy_type = 'float32'
        elif str(dtype) == 'int32':
            node.appy_type = 'int32'
        elif str(dtype) == 'float64':
            node.appy_type = 'float64'
        elif str(dtype) == 'int64':
            node.appy_type = 'int64'
        else:
            assert False
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        types = [child.appy_type for child in [node.left, node.right] if hasattr(child, "appy_type")]
        if all(types):
            node.appy_type = types[0]
        else:
            assert False

    def visit_Name(self, node):
        assert node.id in self.name_to_type, f"Type not found for name {node.id}"
        node.appy_type = self.name_to_type[node.id]
        if self.verbose:
            print("[AttachTypes] assign type to name", node.id, node.appy_type)

    def visit_Assign(self, node):
        self.visit(node.value)
        target = node.targets[0]
        if isinstance(target, ast.Name):
            if target.id not in self.name_to_type:
                self.name_to_type[target.id] = node.value.appy_type
            else:
                assert self.name_to_type[target.id] == node.value.appy_type
        self.visit(target)
    

def visit(tree, val_map):
    visitor = AttachTypes(val_map)
    visitor.visit(tree)