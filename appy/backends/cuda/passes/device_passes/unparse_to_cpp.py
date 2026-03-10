import ast
from ..constants import BLOCK_SIZE


class CppUnparser(ast.NodeVisitor):
    def __init__(self):
        self.code = ""
        self.curent_indent = 4

    def visit_Module(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def visit_Expr(self, node):
        self.visit(node.value)
        self.code += ";\n"

    def visit_Assign(self, node):
        self.code += self.curent_indent * " "
        self.visit(node.targets[0])
        self.code += " = "
        self.visit(node.value)
        self.code += ";\n"

    def visit_Break(self, node):
        self.code += self.curent_indent * " " + "break;\n"

    def visit_Continue(self, node):
        self.code += self.curent_indent * " " + "continue;\n"

    def visit_Name(self, node):
        self.code += node.id

    def visit_BinOp(self, node):
        self.code += "("
        self.visit(node.left)
        ops = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
            ast.FloorDiv: "/", ast.Mod: "%", ast.Pow: "**",
            ast.LShift: "<<", ast.RShift: ">>",
            ast.BitOr: "|", ast.BitAnd: "&", ast.BitXor: "^",
        }
        op_type = type(node.op)
        if op_type not in ops:
            raise NotImplementedError(f"Unsupported binary operator: {op_type}")
        self.code += f" {ops[op_type]} "
        self.visit(node.right)
        self.code += ")"

    def visit_BoolOp(self, node):
        ops = {ast.And: "&&", ast.Or: "||"}
        self.code += "("
        for i, value in enumerate(node.values):
            self.visit(value)
            if i != len(node.values) - 1:
                self.code += " " + ops[type(node.op)] + " "
        self.code += ")"

    def visit_UnaryOp(self, node):
        ops = {ast.USub: "-", ast.UAdd: "+", ast.Not: "!", ast.Invert: "~"}
        op_type = type(node.op)
        self.code += ops.get(op_type, "")
        self.visit(node.operand)

    def visit_Call(self, node):
        self.visit(node.func)
        self.code += "("
        for i, arg in enumerate(node.args):
            self.visit(arg)
            if i != len(node.args) - 1:
                self.code += ", "
        self.code += ")"

    def visit_If(self, node):
        self.code += self.curent_indent * " " + "if ("
        self.visit(node.test)
        self.code += ") {\n"
        self.curent_indent += 4
        for stmt in node.body:
            self.visit(stmt)
        self.curent_indent -= 4
        self.code += self.curent_indent * " " + "}"
        if node.orelse:
            self.code += " else {\n"
            self.curent_indent += 4
            for stmt in node.orelse:
                self.visit(stmt)
            self.curent_indent -= 4
            self.code += self.curent_indent * " " + "}"
        self.code += "\n"

    def visit_While(self, node):
        self.code += self.curent_indent * " " + "while ("
        self.visit(node.test)
        self.code += ") {\n"
        self.curent_indent += 4
        for stmt in node.body:
            self.visit(stmt)
        self.curent_indent -= 4
        self.code += self.curent_indent * " " + "}\n"

    def visit_For(self, node):
        assert isinstance(node.iter, ast.Call) and ast.unparse(node.iter.func) == "range", \
            "Only for-range loops are supported"
        range_args = node.iter.args
        assert len(range_args) in (1, 2, 3)
        if len(range_args) == 1:
            start, end, step = "0", ast.unparse(range_args[0]), "1"
        elif len(range_args) == 2:
            start, end, step = ast.unparse(range_args[0]), ast.unparse(range_args[1]), "1"
        else:
            start, end, step = (ast.unparse(range_args[0]),
                                 ast.unparse(range_args[1]),
                                 ast.unparse(range_args[2]))
        target = node.target.id
        self.code += self.curent_indent * " "
        self.code += f"for (int {target} = {start}; {target} < {end}; {target} += {step})" + " {\n"
        self.curent_indent += 4
        for stmt in node.body:
            self.visit(stmt)
        self.curent_indent -= 4
        self.code += self.curent_indent * " " + "}\n"

    def visit_Compare(self, node):
        self.code += "("
        self.visit(node.left)
        ops = {
            ast.Eq: "==", ast.NotEq: "!=",
            ast.Lt: "<", ast.LtE: "<=",
            ast.Gt: ">", ast.GtE: ">=",
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

    def visit_Attribute(self, node):
        # e.g. atomicAdd, etc. — emit as plain name using the attribute
        self.code += node.attr


class CudaCppUnparser(CppUnparser):
    _CUDA_SPECIALS = {
        '__cuda_shared_mem_decl',
        '__cuda_shared_reduce',
        '__cuda_atomic_add',
    }

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in self._CUDA_SPECIALS):
            self._visit_cuda_special(node.value)
        else:
            super().visit_Expr(node)

    def _visit_cuda_special(self, call):
        indent = self.curent_indent * ' '
        name = call.func.id
        if name == '__cuda_shared_mem_decl':
            sh_var = call.args[0].value
            ty = call.args[1].value
            size = call.args[2].value
            self.code += f'{indent}__shared__ {ty} {sh_var}[{size}];\n'
        elif name == '__cuda_shared_reduce':
            var = call.args[0].value
            sh_var = call.args[1].value
            op = call.args[2].value
            self._emit_shared_reduce(var, sh_var, op, indent)
        elif name == '__cuda_atomic_add':
            ptr = call.args[0].value
            val = call.args[1].value
            self.code += f'{indent}atomicAdd(&{ptr}, {val});\n'

    def _emit_shared_reduce(self, var, sh_var, op, indent):
        if op == 'sum':
            reduce_stmt = f'{sh_var}[lane] += {sh_var}[lane + stride]'
        elif op == 'min':
            reduce_stmt = f'{sh_var}[lane] = min({sh_var}[lane], {sh_var}[lane + stride])'
        elif op == 'max':
            reduce_stmt = f'{sh_var}[lane] = max({sh_var}[lane], {sh_var}[lane + stride])'
        else:
            raise NotImplementedError(f'Unsupported reduction op: {op}')
        self.code += (
            f'{indent}{sh_var}[lane] = {var};\n'
            f'{indent}__syncthreads();\n'
            f'{indent}for (int stride = {BLOCK_SIZE // 2}; stride > 0; stride >>= 1) {{\n'
            f'{indent}    if (lane < stride) {reduce_stmt};\n'
            f'{indent}    __syncthreads();\n'
            f'{indent}}}\n'
            f'{indent}{var} = {sh_var}[0];\n'
        )


def visit(tree):
    visitor = CudaCppUnparser()
    visitor.visit(tree)
    return visitor.code
