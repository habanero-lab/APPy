import ast
from . import device_func_types
from . import type_map

class AttachTypes(ast.NodeVisitor):
    def __init__(self, val_map):
        self.val_map = val_map
        self.name_to_type = {}
        self.verbose = False

    def visit_For(self, node):
        # loop index variable defauts to int type
        self.name_to_type[node.target.id] = 'int'
        for child in node.body:
            self.visit(child)
        return node

    def visit_Call(self, node):
        func = None
        if isinstance(node.func, ast.Attribute):
            func = node.func.attr
        elif isinstance(node.func, ast.Name):
            func = node.func.id

        assert hasattr(device_func_types, func), "Unknown device function: " + func
        arg_count = getattr(device_func_types, func).__code__.co_argcount
        assert len(node.args) == arg_count, f"Function {func} takes {arg_count} arguments, but got {len(node.args)}"

        arg_types = []
        for arg in node.args:
            self.visit(arg)
            arg_types.append(arg.appy_type)
        # Assume the return value has the same type as the first argument
        node.appy_type = getattr(device_func_types, func)(*arg_types)

    def visit_Constant(self, node):
        val = node.value
        node.appy_type = type_map.get_metal_type(val)
    
    def visit_Subscript(self, node):
        # Array variables are not visited
        if isinstance(node.value, ast.Name) and node.value.id in self.val_map:
            val = self.val_map[node.value.id]
            assert hasattr(val, "dtype"), f"Non-array found: {node.value.id}, a variable needs to be either an array or a vector type to be subscripted"            
            node.appy_type = type_map.get_metal_type(val)
        elif isinstance(node.value, ast.Name) and node.value.id in self.name_to_type:
            base_type = self.name_to_type[node.value.id]
            assert base_type[-1] in ['2', '3', '4'], f"Unknown vector type: {base_type}"
            node.appy_type = base_type[:-1]
        else:
            assert False, f"Unknown subscript: {node}"

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.generic_visit(node)
        node.appy_type = node.operand.appy_type
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        types = [child.appy_type for child in [node.left, node.right] if hasattr(child, "appy_type")]
        types.sort()
        if types[0] == types[1]:
            node.appy_type = types[0]
        elif types == ["float", "int"]:
            node.appy_type = types[0]
        elif types[1] == types[0] + "2" or types[1] == types[0] + "3" or types[1] == types[0] + "4":
            node.appy_type = types[1]
        else:
            assert False, "Incompatible types for binary oprator: " + str(types)

    def visit_Name(self, node):
        # Must be a scalar here since array variables are not visited (not needed for kernel codegen var decls)
        if node.id in self.val_map:
            node.appy_type = type_map.get_metal_type(self.val_map[node.id])
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
                assert self.name_to_type[target.id] == node.value.appy_type, f"{self.name_to_type[target.id]} != {node.value.appy_type}; var: {target.id}"
        self.visit(target)
    

def visit(tree, val_map):
    '''
    Attach a metal data type to each AST node. The following scalar types are supported:
    * bool
    * int8_t (char)
    * uint8_t (uchar)
    * int16_t (short)
    * uint16_t (ushort)
    * int32_t (int)
    * uint32_t (uint)
    * float (32 bits)

    Three vector types are also supported: a vector of 2, 3 or 4 scalar elements shown above.
    The vector type name has the suffix "2", "3" or "4", e.g. float2, int3 etc. The full list is 
    as follows:
    * booln
    * charn
    * ucharn
    * shortn
    * ushortn
    * intn
    * uintn
    * floatn
    where n is 2, 3 or 4.

    If an AST node cannot be determined to have one of the above types, an unsupported type error 
    is thrown.
    '''
    visitor = AttachTypes(val_map)
    visitor.visit(tree)