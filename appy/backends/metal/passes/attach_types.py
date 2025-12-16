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
            arg_types.append(arg.metal_type)
        # Assume the return value has the same type as the first argument
        node.metal_type = getattr(device_func_types, func)(*arg_types)

    def visit_Constant(self, node):
        val = node.value
        node.metal_type = type_map.get_metal_type(val)
    
    def visit_Subscript(self, node):
        # Array variables are not visited
        if isinstance(node.value, ast.Name) and node.value.id in self.val_map:
            val = self.val_map[node.value.id]
            assert hasattr(val, "dtype"), f"Non-array found: {node.value.id}, a variable needs to be either an array or a vector type to be subscripted"            
            node.metal_type = type_map.get_metal_type(val)
        elif isinstance(node.value, ast.Name) and node.value.id in self.name_to_type:
            base_type = self.name_to_type[node.value.id]
            assert base_type[-1] in ['2', '3', '4'], f"Unknown vector type: {base_type}"
            node.metal_type = base_type[:-1]
        else:
            assert False, f"Unknown subscript: {node}"

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.generic_visit(node)
        node.metal_type = node.operand.metal_type
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        types = [child.metal_type for child in [node.left, node.right]]
        types.sort()
        if types[0] == types[1]:
            node.metal_type = types[0]
            # Special div semantic: int / int -> float
            # In Python, / is always float while in Metal, / is int if both operands are int
            if isinstance(node.op, ast.Div) and types[0] == "int":
                node.metal_type = "float"

        elif types[0] == 'float' and types[1] in ['bool', 'char', 'uchar', 'short', 'ushort', 'int', 'uint']:
            node.metal_type = 'float'
        elif types[1] == 'float' and types[0] in ['bool', 'char', 'uchar', 'short', 'ushort', 'int', 'uint']:
            node.metal_type = 'float'
        elif types[1] == types[0] + "2" or types[1] == types[0] + "3" or types[1] == types[0] + "4":
            node.metal_type = types[1]
        else:
            assert False, "Incompatible types for binary oprator: " + str(types)

    def visit_Name(self, node):
        # Must be a scalar here since array variables are not visited (not needed for kernel codegen var decls)
        if node.id in self.val_map:
            node.metal_type = type_map.get_metal_type(self.val_map[node.id])
        elif node.id in self.name_to_type:
            node.metal_type = self.name_to_type[node.id]
        else:
            node.metal_type = None
            
        if self.verbose:
            print("[AttachTypes] assign type to name", node.id, node.metal_type)

    def bind_type_to_name(self, var, ty):
        self.name_to_type[var] = ty

    def visit_Assign(self, node):
        self.generic_visit(node)
        target = node.targets[0]
        value = node.value
        if target.metal_type != value.metal_type:
            if isinstance(target, ast.Name) and target.metal_type is None:
                # Bind type to name
                self.bind_type_to_name(target.id, value.metal_type)
                target.metal_type = value.metal_type
            # Check type compatibility
            elif type_map.is_arithmetic_scalar_metal_type(target.metal_type) and \
                type_map.is_arithmetic_scalar_metal_type(value.metal_type):
                # Implicit conversions between arithmetic types are defined both in NumPy and Metal
                pass
            else:
                # Type mismatch, throw an exception
                raise TypeError(f"Type mismatch: {target.metal_type} != {value.metal_type}")

        assert target.metal_type is not None and target.metal_type == value.metal_type

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