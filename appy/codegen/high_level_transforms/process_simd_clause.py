'''
This pass converts the `simd` clause to `block` clause.
'''
import ast
from .utils import dict_to_pragma

class ProcessSimdClause(ast.NodeTransformer):
    def visit_For(self, node: ast.For):
        if hasattr(node, 'pragma'):
            pragma_dict = node.pragma_dict
            if 'simd' in pragma_dict:
                pragma_dict['block'] = pragma_dict['simd']
                del pragma_dict['simd']

                # If `block` has no value associated (by default it's True), set it to 256 or 1024
                if pragma_dict['block'] is True:
                    if 'reduction' in pragma_dict:
                        pragma_dict['block'] = 1024
                    else:
                        pragma_dict['block'] = 256
            node.pragma_dict = pragma_dict
            node.pragma = dict_to_pragma(pragma_dict)
        return node
    

def transform(node):
    visitor = ProcessSimdClause()
    return visitor.visit(node)