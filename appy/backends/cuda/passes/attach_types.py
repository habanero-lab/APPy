import ast
from . import device_func_types
from . import type_map


class AttachTypes(ast.NodeVisitor):
    def __init__(self, val_map):
        self.val_map = val_map
        self.name_to_type = {}
        self.verbose = False

    def visit_For(self, node):
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
        assert len(node.args) == arg_count, \
            f"Function {func} takes {arg_count} arguments, but got {len(node.args)}"

        arg_types = []
        for arg in node.args:
            self.visit(arg)
            arg_types.append(arg.cuda_type)
        node.cuda_type = getattr(device_func_types, func)(*arg_types)

    def visit_Constant(self, node):
        node.cuda_type = type_map.get_cuda_type(node.value)

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.val_map:
            val = self.val_map[node.value.id]
            assert hasattr(val, "dtype"), \
                f"Non-array found: {node.value.id}, variable needs to be an array to be subscripted"
            node.cuda_type = type_map.get_cuda_type(val)
        else:
            assert False, f"Unknown subscript: {node}"

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        node.cuda_type = node.operand.cuda_type

    def visit_BinOp(self, node):
        self.generic_visit(node)
        types = [child.cuda_type for child in [node.left, node.right]]
        types_sorted = sorted(types)
        int_types = {'bool', 'int8_t', 'uint8_t', 'int16_t', 'uint16_t', 'int', 'unsigned int'}
        if isinstance(node.op, ast.Div) and types[0] in int_types and types[1] in int_types:
            # Python semantics: int / int => float
            node.cuda_type = 'float'
        elif types[0] == types[1]:
            node.cuda_type = types[0]
        elif 'double' in types:
            node.cuda_type = 'double'
        elif 'float' in types:
            node.cuda_type = 'float'
        else:
            node.cuda_type = types_sorted[-1]

    def visit_Name(self, node):
        if node.id in self.val_map:
            node.cuda_type = type_map.get_cuda_type(self.val_map[node.id])
        elif node.id in self.name_to_type:
            node.cuda_type = self.name_to_type[node.id]
        else:
            node.cuda_type = None

    def bind_type_to_name(self, var, ty):
        self.name_to_type[var] = ty

    def visit_Assign(self, node):
        self.generic_visit(node)
        target = node.targets[0]
        value = node.value
        if target.cuda_type != value.cuda_type:
            if isinstance(target, ast.Name) and target.cuda_type is None:
                self.bind_type_to_name(target.id, value.cuda_type)
                target.cuda_type = value.cuda_type
            elif type_map.is_arithmetic_scalar_cuda_type(target.cuda_type) and \
                    type_map.is_arithmetic_scalar_cuda_type(value.cuda_type):
                pass  # implicit arithmetic conversions are fine
            else:
                raise TypeError(f"Type mismatch: {target.cuda_type} != {value.cuda_type}")


def visit(tree, val_map):
    '''
    Attach a CUDA C data type to each AST node. Supported scalar types:
      bool, int8_t, uint8_t, int16_t, uint16_t, int, unsigned int, float, double
    '''
    visitor = AttachTypes(val_map)
    visitor.visit(tree)
