import ast

class ApplyMaskToReduction(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.value.id == "tl":
            funcname = node.func.attr
            if funcname in ["sum", "min", "max"]:
                assert len(node.args) == 1

            arg0 = node.args[0]
            if hasattr(arg0, "mask"):
                mask = ast.parse(arg0.mask).body[0].value
                if funcname == "sum":
                    # Update the argument to be `tl.where(mask, arg0, 0)`
                    node.args[0] = ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='tl', ctx=ast.Load()),
                            attr='where'
                        ),
                        args=[mask, arg0, ast.Constant(value=0)],
                        keywords=[]
                    )
                elif funcname == "min":
                    # Update the argument to be `tl.where(mask, arg0, float("inf"))`
                    node.args[0] = ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='tl', ctx=ast.Load()),
                            attr='where'
                        ),
                        args=[mask, arg0, ast.Call(
                            func=ast.Name(id='float', ctx=ast.Load()),
                            args=[ast.Constant(value="inf")],
                            keywords=[]
                        )],
                        keywords=[]
                    )
                elif funcname == "max":
                    # Update the argument to be `tl.where(mask, arg0, float("-inf"))`
                    node.args[0] = ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='tl', ctx=ast.Load()),
                            attr='where'
                        ),
                        args=[mask, arg0, ast.Call(
                            func=ast.Name(id='float', ctx=ast.Load()),
                            args=[ast.Constant(value="-inf")],
                            keywords=[] 
                        )],
                        keywords=[]
                    )

        return node
        

def transform(tree):
    return ApplyMaskToReduction().visit(tree)