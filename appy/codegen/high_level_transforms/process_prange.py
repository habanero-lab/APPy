'''
This pass inspects for loops that are marked as a prange, and converts them to
regular for-range loops with the `parallel for` pragma. 
It also automatically detects any innermost loops that could be annotated
with the `simd` pragma.
'''
import ast
from .utils import dict_to_pragma, parse_pragma


class VisitReadWriteArrays(ast.NodeVisitor):
    def __init__(self):
        # Map from array name to a list of appeared indices
        self.read_tensors = {}
        self.write_tensors = {}

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name):
            name = node.value.id
            ss = ast.unparse(node.slice)
            if isinstance(node.ctx, ast.Load):                                
                if name not in self.read_tensors:
                    self.read_tensors[name] = []
                if ss not in self.read_tensors[name]:
                    self.read_tensors[name].append(ss)
            elif isinstance(node.ctx, ast.Store):
                #self.write_tensors.add(node.value.id)
                if name not in self.write_tensors:
                    self.write_tensors[name] = []
                if ss not in self.write_tensors[name]:
                    self.write_tensors[name].append(ss)


class VectorizeInnerMostLoops(ast.NodeTransformer):
    def visit_For(self, node):
        self.generic_visit(node)
        vectorizable = True
        for stmt in node.body:
            # If any non-assignment node is found, then it is not vectorizable
            if not isinstance(stmt, ast.Assign):
                vectorizable = False
                break

        array_visitor = VisitReadWriteArrays()
        array_visitor.visit(node)
        # If the same array is both read and written, and the indices are different, 
        # it is considered not vectorizable
        for t in array_visitor.write_tensors:
            if t in array_visitor.read_tensors:
                if array_visitor.write_tensors[t] != array_visitor.read_tensors[t]:
                    vectorizable = False
                    break

        if vectorizable:
            if not hasattr(node, 'pragma'):
                node.pragma = '#pragma simd'
                node.pragma_dict = parse_pragma(node.pragma)
            else:
                # Append simd pragma to existing pragma
                node.pragma = node.pragma + ' simd'
                node.pragma_dict = parse_pragma(node.pragma)
        return node
    

class ProcessPrange(ast.NodeTransformer):
    def visit_For(self, node):
        # Both prange and appy.prange are supported, first convert appy.prange to prange
        if ast.unparse(node.iter.func) == 'appy.prange':
            node.iter.func = ast.Name(id='prange', ctx=ast.Load())
        
        is_prange = node.iter.func.id == 'prange'
        if not is_prange:
            self.generic_visit(node)
            return node
        
        assert not hasattr(node, 'pragma')

        node.pragma = '#pragma parallel for'
        node.pragma_dict = parse_pragma(node.pragma)
        node.iter.func.id = 'range'

        vectorizer = VectorizeInnerMostLoops()
        node = vectorizer.visit(node)
        return node

        
def transform(node):
    return ProcessPrange().visit(node)