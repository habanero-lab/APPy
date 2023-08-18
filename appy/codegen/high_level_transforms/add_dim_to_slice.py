import ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy

# class AddDimToSlice(ast.NodeTransformer):
#     def __init__(self, dim_info):
#         self.dim_info = dim_info

#     def visit_Subscript(self, node):
#         assert isinstance(node.value, ast.Name)
#         elts = None
#         if node.slice.__class__.__name__ == 'Tuple':
#             elts = node.slice.elts
#         else:
#             elts = [node.slice]
        
#         for 


class AddDimToSlice(ast.NodeTransformer):
    def __init__(self, dim_info):
        self.dim_info = dim_info

    def visit_Subscript(self, node):
        
        if isinstance(node.value, ast.Name) and node.slice.__class__.__name__ == 'Tuple':
            array_name = node.value.id
            dims = self.dim_info.get(array_name, ())
            new_slice_elements = []

            for dim_slice, dim_size in zip(node.slice.elts, dims):
                
                if isinstance(dim_slice, ast.Slice):
                    lower = dim_slice.lower
                    upper = dim_slice.upper
                    step = dim_slice.step

                    if upper is None:
                        new_upper = ast.Name(id=dim_size, ctx=ast.Load())
                    elif isinstance(upper, ast.Constant):
                        new_upper = ast.BinOp(left=ast.Name(id=dim_size, ctx=ast.Load()), 
                                              op=ast.Add(), 
                                              right=upper)
                    elif isinstance(upper, ast.UnaryOp) and isinstance(upper.operand, ast.Constant):
                        new_upper = ast.BinOp(left=ast.Name(id=dim_size, ctx=ast.Load()), 
                                              op=ast.Add(), 
                                              right=upper)
                    else:
                        new_upper = upper

                    import sympy
                    new_upper_simplified = str(sympy.simplify(unparse(new_upper).replace('N', 'n'))).replace('n', 'N')
                    new_upper = to_ast_expr(new_upper_simplified)
                    new_dim_slice = ast.Slice(lower, new_upper, step) 
                                       
                    new_slice_elements.append(new_dim_slice)
                else:
                    new_slice_elements.append(dim_slice)

            new_slice = ast.Tuple(elts=new_slice_elements, ctx=ast.Load())
            new_subscript = ast.Subscript(value=node.value, slice=new_slice, ctx=node.ctx)
            return new_subscript

        return node
