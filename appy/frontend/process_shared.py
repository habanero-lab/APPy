import re
import ast_comments as ast

class ReplaceNameWithSubscript(ast.NodeTransformer):
    def __init__(self, scalars):
        self.scalars = scalars

    def visit_Name(self, node):
        if node.id in self.scalars:
            node = ast.Subscript(
                value=ast.Name(id=node.id, ctx=ast.Load()),
                slice=ast.Constant(value=0),
                ctx=node.ctx
            )
        return node

class ProcessSharedPragmas(ast.NodeTransformer):
    def __init__(self):
        self.pragma = None

    def visit_Comment(self, node: ast.Comment):
        if node.value.startswith("#pragma parallel for"):
            self.pragma = node.value
            return None  # remove the recorded pragma
        return node
    
    def visit_For(self, node):
        if self.pragma:
            m = re.search(r"shared\(([^)]*)\)", self.pragma)
            if m:
                shared_vars = [v.strip() for v in m.group(1).split(',')]
                # Rewrite the shared variables to np.array, e.g. x = np.array(x)
                pre_loop_code_str = "\n".join(
                    [
                        f"{x} = numpy.array({x})" for x in shared_vars
                    ]
                )

                # Convert back using .item()
                post_loop_code_str = "\n".join(
                    [
                        f"{x} = {x}.item()" for x in shared_vars
                    ]
                )
                newnode = ReplaceNameWithSubscript(scalars=shared_vars).visit(node)
                loop_source = "\n".join([pre_loop_code_str, self.pragma, ast.unparse(newnode), post_loop_code_str])
            else:
                loop_source = self.pragma + "\n" + ast.unparse(node)
            self.pragma = None  # reset
            node = ast.parse(loop_source).body
            return node
        self.generic_visit(node)
        return node

    

def transform(tree):
    '''
    Rewrite variables in the shared directive into array form. For example,

    #pragma parallel for shared(s)
    for i in range(10):
        s += 1

    becomes

    s = np.array(s)
    #pragma parallel for shared(s)
    for i in range(10):
        s[i] += 1
    s = s.item()
    '''
    return ProcessSharedPragmas().visit(tree)