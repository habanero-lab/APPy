import ast
from . import metal_math

class AttachTypes(ast.NodeVisitor):
    def __init__(self, val_map):
        self.val_map = val_map
        self.name_to_type = {}
        self.py_to_cpp = {
            # Python types
            'int': 'int',
            'float': 'float',

            # NumPy dtypes
            'float32': 'float',
            'int32': 'int',
            'uint8': 'uint8_t'
        }
        self.verbose = True

    def visit_For(self, node):
        # loop index variable defauts to int type
        self.name_to_type[node.target.id] = 'int'
        for child in node.body:
            self.visit(child)
        return node

    def visit_Call(self, node):
        # Handle special built-in functions
        if isinstance(node.func, ast.Name):
            if node.func.id == "int":
                node.appy_type = "int"
                return
            elif node.func.id == "float":
                node.appy_type = "float"
                return

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
        node.appy_type = self.py_to_cpp[type(val).__name__]
    
    def visit_Subscript(self, node):
        # Array variables are not visited
        assert isinstance(node.value, ast.Name) and node.value.id in self.val_map
        dtype = self.val_map[node.value.id].dtype
        node.appy_type = self.py_to_cpp[str(dtype)]
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        types = [child.appy_type for child in [node.left, node.right] if hasattr(child, "appy_type")]
        if types[0] == types[1]:
            node.appy_type = types[0]
        elif set(types) == {"float", "int"}:
            node.appy_type = "float"
        else:
            assert False, "Incompatible types for binary oprator: " + str(types)

    def visit_Name(self, node):
        # Must be a scalar here since array variables are not visited (not needed for kernel codegen var decls)
        if node.id in self.val_map:
            node.appy_type = self.py_to_cpp[type(self.val_map[node.id]).__name__]
        else:
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