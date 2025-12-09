import ast

class CppUnparser(ast.NodeVisitor):
    def __init__(self):
        self.code = ""

    def visit_Module(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def visit_Expr(self, node):
        # For expressions like function calls
        self.visit(node.value)
        self.code += ";\n"  # C++ style line ending

    def visit_Assign(self, node):
        # Assume single target for simplicity
        target = node.targets[0]
        self.visit(target)
        self.code += " = "
        self.visit(node.value)
        self.code += ";"  # Add semicolon at end of assignment
        self.code += "\n"

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
        self.code += "if ("
        self.visit(node.test)
        self.code += ") {\n"
        for stmt in node.body:
            self.visit(stmt)
        self.code += "}"
        if node.orelse:
            self.code += " else {\n"
            for stmt in node.orelse:
                self.visit(stmt)
            self.code += "}"
        self.code += "\n"

    def visit_While(self, node):
        self.code += "while ("
        self.visit(node.test)
        self.code += ") {\n"
        for stmt in node.body:
            self.visit(stmt)
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


def visit(tree):
    visitor = CppUnparser()
    visitor.visit(tree)
    return visitor.code
