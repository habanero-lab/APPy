'''
This pass rewrites array lib calls inside prange regions into the corresponding Triton lib calls.
'''

import ast

class RewriteLibCalls(ast.NodeTransformer):
    '''
    Maintain a list of recognized built-in lib functions, and 
    rewrite them to their Triton equivalents. Both the raw call, e.g. `log`,
    and the attribute call, e.g. `np.log`, forms are converted.
    '''
    def __init__(self):
        self.lib_funcs = {
            'abs': 'tl.abs',
            'cdiv': 'tl.cdiv',
            'ceil': 'tl.ceil',
            'clamp': 'tl.clamp',
            'cos': 'tl.cos',
            'cosh': 'tl.cosh',
            'exp': 'tl.exp',
            'erf': 'tl.erf',
            'floor': 'tl.floor',
            'log': 'tl.log',
            'log2': 'tl.log2',
            'maximum': 'tl.maximum',
            'minimum': 'tl.minimum',
            'rsqrt': 'tl.rsqrt',
            'sin': 'tl.sin',
            'sinh': 'tl.sinh',
            'sqrt': 'tl.sqrt',
            'tan': 'tl.tan',

            # Reduction operations
            'sum': 'tl.sum',
            'max': 'tl.max',
            'min': 'tl.min',
            'argmax': 'tl.argmax',
            'argmin': 'tl.argmin',

            # Indexing operations
            'where': 'tl.where',
            'flip': 'tl.flip',
        }

    def visit_Call(self, node):
        '''
        Rewrite calls to math functions to their Triton equivalents.
        Examples:
        np.log -> tl.log
        log -> tl.log
        np.maximum -> tl.maximum
        '''
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.lib_funcs:
            # Direct call, e.g. log()
            node = ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr=self.lib_funcs[node.func.id].split('.')[-1], ctx=ast.Load()),
                args=node.args,
                keywords=node.keywords
            )

        elif isinstance(node.func, ast.Attribute):
            # Attribute call, e.g. np.log()
            if node.func.attr in self.lib_funcs:
                node.func.value.id = 'tl'
                node.func.attr = self.lib_funcs[node.func.attr].split('.')[-1]

        return node


def transform(node):
    return RewriteLibCalls().visit(node)