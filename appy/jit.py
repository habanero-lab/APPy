from ast_utils import *
import ast_comments as ast
import inspect
import textwrap

class ExtractParallelLoops(ast.NodeTransformer):
    def __init__(self):
        self.pfor_pragma = None

    def visit_Comment(self, node):
        comment = node.value.strip()
    
        if comment.startswith('#pragma parallel for'):
            self.pfor_pragma = node
            return None
        return node

    def visit_For(self, node: ast.For):
    
        if self.pfor_pragma:
            assign_node = to_ast_node(f'__code_str = """{ast.unparse(self.pfor_pragma) + '\n' + ast.unparse(node)}"""')
            call_node = to_ast_node(f'appy.compile_from_src(__code_str)')
            self.pfor_pragma = None
            return [assign_node, call_node]
        else:
            self.generic_visit(node)
            return node

def jit(func):
    source_code = inspect.getsource(func)
    source_code = textwrap.dedent(source_code)  # Remove indentation
    tree = ast.parse(source_code)
    transformer = ExtractParallelLoops()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    print(ast.unparse(tree))

    # Compile the modified AST to a code object
    filename = inspect.getfile(func)
    code = compile(tree, filename=filename, mode='exec')

    # Create a namespace in which to execute the code object
    namespace = func.__globals__.copy()
    exec(code, namespace)

    # Return the new version of the function
    return namespace[func.__name__]


if __name__ == '__main__':
    import appy
    import numpy as np

    def foo():
        a = np.zeros(10)
        b = np.ones(10)
        #pragma parallel for
        for i in range(a.shape[0]):
            a[i] += b[i]

    newfoo = jit(foo)
    newfoo()
