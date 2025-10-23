import ast

class AttachTypesInner(ast.NodeTransformer):
    def __init__(self, val_map):
        self.type_map = {}
        self.val_map = val_map

    def visit_For(self, node):
        # The loop target has type 'int32'
        node.target.appy_type = 'int32'
        self.type_map[node.target.id] = 'int32'
        self.generic_visit(node)

    def visit_Assign(self, node):
        self.visit(node.value)
        target = node.targets[0]
        target.appy_type = node.value.appy_type
        if isinstance(target, ast.Name):
            self.type_map[target.id] = target.appy_type

    def visit_Name(self, node):
        if node.id in self.type_map:
            node.appy_type = self.type_map[node.id]
        elif node.id in self.val_map:
            import numpy as np
            val = self.val_map[node.id]
            if isinstance(val, int):
                node.appy_type = 'int32'
            elif isinstance(val, float):
                node.appy_type = 'float64'
            elif isinstance(val, np.ndarray):
                if val.dtype == np.int32:
                    node.appy_type = 'int32*'
                elif val.dtype == np.float32:
                    node.appy_type = 'float32*'
                elif val.dtype == np.float64:
                    node.appy_type = 'float64*'
            else:
                raise NotImplementedError(f"Type attachment not implemented for variable {node.id} of type {type(val)}")
        elif node.id == "appy":
            pass # skip this
        else:
            raise NotImplementedError(f"Type not found for name {node.id}")
        return node
    
    def visit_Constant(self, node):
        if isinstance(node.value, int):
            node.appy_type = 'int32'
        elif isinstance(node.value, float):
            node.appy_type = 'float64'
        return node
    
    def visit_Subscript(self, node):        
        self.visit(node.value)
        self.visit(node.slice)
        assert node.slice.appy_type == 'int32', "Only integer indexing is supported"
        # The type of the subscripted value is the base type without '*'
        if node.value.appy_type.endswith('*'):
            dtype = node.value.appy_type[:-1]
            node.appy_type = dtype
        return node
    
    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        # Simple type inference for binary operations
        if node.left.appy_type == node.right.appy_type:
            node.appy_type = node.left.appy_type
        else:
            raise NotImplementedError(f"Type inference not implemented for BinOp with types {node.left.appy_type} and {node.right.appy_type}")
        return node
    
class AttachTypes(ast.NodeTransformer):
    def __init__(self, val_map):
        self.val_map = val_map

    def visit_For(self, node):
        return AttachTypesInner(self.val_map).visit(node)