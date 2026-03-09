import ast
from ..constants import SIMD_WIDTH


class CppUnparser(ast.NodeVisitor):
    def __init__(self):
        self.code = ""
        self.curent_indent = 4

    def visit_Module(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def visit_Expr(self, node):
        # For expressions like function calls
        self.visit(node.value)
        self.code += ";\n"  # C++ style line ending

    def visit_Assign(self, node):
        # Assume single target for simplicity
        self.code += self.curent_indent * " "
        target = node.targets[0]
        self.visit(target)
        self.code += " = "
        self.visit(node.value)
        self.code += ";"  # Add semicolon at end of assignment
        self.code += "\n"

    def visit_Break(self, node):
        self.code += self.curent_indent * " "
        self.code += "break;\n"

    def visit_Continue(self, node):
        self.code += self.curent_indent * " "
        self.code += "continue;\n"

    def visit_Name(self, node):
        self.code += node.id

    def visit_BinOp(self, node):
        self.code += "("
        self.visit(node.left)
        ops = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "/",  # keep as '/' for simplicity
            ast.Mod: "%",
            ast.Pow: "**",      # just output '**' for now
            ast.LShift: "<<",
            ast.RShift: ">>",
            ast.BitOr: "|",
            ast.BitAnd: "&",
            ast.BitXor: "^",
        }
        op_type = type(node.op)
        if op_type in ops:
            self.code += f" {ops[op_type]} "
        else:
            raise NotImplementedError(f"Unsupported binary operator: {op_type}")
        self.visit(node.right)
        self.code += ")"

    def visit_BoolOp(self, node):
        self.code += "("
        ops = {
            ast.And: "&&",
            ast.Or: "||",
        }
        for i, value in enumerate(node.values):
            self.visit(value)
            if i != len(node.values) - 1:
                self.code += " " + ops[type(node.op)] + " "
        self.code += ")"

    def visit_Call(self, node):
        self.visit(node.func)
        self.code += "("
        for i, arg in enumerate(node.args):
            self.visit(arg)
            if i != len(node.args) - 1:
                self.code += ", "
        self.code += ")"

    def visit_If(self, node):
        self.code += self.curent_indent * " "
        self.code += "if ("
        self.visit(node.test)
        self.code += ") {\n"
        self.curent_indent += 4
        for stmt in node.body:
            self.visit(stmt)
        self.curent_indent -= 4
        self.code += self.curent_indent * " "
        self.code += "}"
        if node.orelse:
            self.code += " else {\n"
            for stmt in node.orelse:
                self.visit(stmt)
            self.code += "}"
        self.code += "\n"

    def visit_While(self, node):
        self.code += self.curent_indent * " "
        self.code += "while ("
        self.curent_indent += 4
        self.visit(node.test)
        self.code += ") {\n"
        for stmt in node.body:
            self.visit(stmt)
        self.curent_indent -= 4
        self.code += self.curent_indent * " "
        self.code += "}\n"

    def visit_For(self, node: ast.For):
        # Only for-range loops are supported
        assert isinstance(node.iter, ast.Call) and ast.unparse(node.iter.func) == "range", "Only for-range loops are supported"
        range_args = node.iter.args
        assert len(range_args) in (1, 2, 3), \
            f"for-range loops must have 1, 2, or 3 arguments, got: {ast.unparse(node.iter)}"
        if len(range_args) == 1:
            start, end, step = "0", ast.unparse(range_args[0]), "1"
        elif len(range_args) == 2:
            start, end, step = ast.unparse(range_args[0]), ast.unparse(range_args[1]), "1"
        else:
            start, end, step = ast.unparse(range_args[0]), ast.unparse(range_args[1]), ast.unparse(range_args[2])
        target = node.target.id
        self.code += self.curent_indent * " "
        self.code += f"for (int {target} = {start}; {target} < {end}; {target} += {step}) " + "{\n"
        self.curent_indent += 4
        for stmt in node.body:
            self.visit(stmt)
        self.curent_indent -= 4
        self.code += self.curent_indent * " "
        self.code += "}\n"

    def visit_Compare(self, node):
        self.code += "("
        self.visit(node.left)
        ops = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        for op, comparator in zip(node.ops, node.comparators):
            self.code += " " + ops[type(op)] + " "
            self.visit(comparator)
        self.code += ")"

    def visit_Subscript(self, node):
        self.visit(node.value)
        self.code += "["
        self.visit(node.slice)
        self.code += "]"

    def visit_Constant(self, node):
        self.code += str(node.value)
        if isinstance(node.value, float):
            self.code += "f"


class MetalCppUnparser(CppUnparser):
    _METAL_SPECIALS = {
        '__metal_shared_mem_decl',
        '__metal_threadgroup_barrier',
        '__metal_tree_reduce',
    }

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in self._METAL_SPECIALS):
            self._visit_metal_special(node.value)
        else:
            super().visit_Expr(node)

    def _visit_metal_special(self, call):
        indent = self.curent_indent * ' '
        name = call.func.id
        if name == '__metal_shared_mem_decl':
            tg_var = call.args[0].value
            ty = call.args[1].value
            size = call.args[2].value
            self.code += f'{indent}threadgroup {ty} {tg_var}[{size}];\n'
        elif name == '__metal_threadgroup_barrier':
            self.code += f'{indent}threadgroup_barrier(mem_flags::mem_threadgroup);\n'
        elif name == '__metal_tree_reduce':
            tg_var = call.args[0].value
            op = call.args[1].value
            self._emit_tree_reduce(tg_var, op, indent)

    def _emit_tree_reduce(self, tg_var, op, indent):
        if op == 'sum':
            reduce_stmt = f'{tg_var}[lane] += {tg_var}[lane + stride]'
        elif op == 'min':
            reduce_stmt = f'{tg_var}[lane] = min({tg_var}[lane], {tg_var}[lane + stride])'
        elif op == 'max':
            reduce_stmt = f'{tg_var}[lane] = max({tg_var}[lane], {tg_var}[lane + stride])'
        else:
            raise NotImplementedError(f'Unsupported reduction op: {op}')
        self.code += (
            f'{indent}for (uint stride = {SIMD_WIDTH // 2}; stride > 0; stride >>= 1) {{\n'
            f'{indent}    if (lane < stride) {reduce_stmt};\n'
            f'{indent}    threadgroup_barrier(mem_flags::mem_threadgroup);\n'
            f'{indent}}}\n'
        )


def visit(tree):
    visitor = MetalCppUnparser()
    visitor.visit(tree)
    return visitor.code
