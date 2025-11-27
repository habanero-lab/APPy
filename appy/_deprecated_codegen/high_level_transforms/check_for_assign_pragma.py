from ast import unparse
from appy.ast_utils import *
from .utils import *


def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()


class CheckForSlice(ast.NodeVisitor):
    def __init__(self):
        self.slices = []

    def visit_Slice(self, node: ast.Slice):
        if unparse(node) not in [unparse(x) for x in self.slices]:
            self.slices.append(node)


class InspectAssign(ast.NodeVisitor):
    def __init__(self, arg_vals) -> None:
        self.arg_vals = arg_vals

    def visit_Assign(self, node):
        slice_checker = CheckForSlice()
        slice_checker.visit(node)
        slices = slice_checker.slices

        if len(slices) == 0:
            return
        elif len(slices) == 1:
            # Automatically add pragma for the slice if it doesn't have any
            if not has_tensor_pragma(node):
                s = slices[0]
                pragma = f'#pragma {unparse(s)}=>block({get_default_block_size()})'
                # Apply the small dimension optimization if the slice is a const argument
                if s.lower == None and unparse(s.upper) in self.arg_vals:
                    runtime_val = self.arg_vals[unparse(s.upper)]
                    if runtime_val <= 2048:
                        pragma = f'#pragma {unparse(s)}=>block({next_power_of_2(runtime_val)}),single_block'
                                        
                if has_atomic_pragma(node):
                    pragma = pragma.replace('#pragma', '#pragma atomic')
                #print(pragma)
                node.pragma = pragma
                return
        else:
            if not has_tensor_pragma(node):
                assert False, f"Please specify a pragma for slice: {unparse(node)}"
           

class CheckAssignPragma(ast.NodeTransformer):
    def __init__(self, arg_vals) -> None:
        self.arg_vals = arg_vals

    def visit_For(self, node: ast.For):    
        if hasattr(node, 'pragma'):
            visitor = InspectAssign(self.arg_vals)
            visitor.visit(node)
        return node