import ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy

class AppendBarrierToWrite(ast.NodeTransformer):
    # def visit_For(self, node: ast.For):
        # newbody = []
        # for s in node.body:
        #     newbody.append(s)
        #     if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Subscript):
        #         newbody += [to_ast_node('tl.debug_barrier()')]
            
        #     if isinstance(s, ast.For):
        #         if hasattr(s, 'from_tensor_expr'):
        #             newbody += [to_ast_node('tl.debug_barrier()')]
        #         else:

        # node.body = newbody
        # return node

    def visit_Assign(self, node: ast.Assign):        
        if isinstance(node.targets[0], ast.Subscript) and not hasattr(node, 'no_sync'):
            #print('to insert after write')
            #dump(node)
            return node, to_ast_node('tl.debug_barrier()')
        else:
            return node

class RemoveBarrierCall(ast.NodeTransformer):
    def __init__(self, removed):
        self.removed = removed

    def visit_Expr(self, node: ast.Call):        
        if not unparse(node).startswith('tl.debug_barrier'):
            return node
        else:
            self.removed.append(node)

class InsertBarrier(ast.NodeTransformer):
    def visit_For(self, node: ast.For):
        if not hasattr(node, 'pragma'):
            self.generic_visit(node)            
        else:
            assert '#pragma parallel' in node.pragma
            AppendBarrierToWrite().visit(node)
        return node

class RemoveBarrierInsideTE(ast.NodeTransformer):
    def visit_For(self, node: ast.For):
        if hasattr(node, 'from_tensor_expr'):
            removed = []
            RemoveBarrierCall(removed).visit(node)
            if len(removed) == 0:
                return node
            elif hasattr(node, 'pragma') and '#pragma parallel' in node.pragma:
                return node
            else:
                return [node, to_ast_node('tl.debug_barrier()')]
                #return node
        else:
            self.generic_visit(node)
            return node


# class InsertBarrierOld(ast.NodeTransformer):
#     def visit_For(self, node: ast.For):
#         if not hasattr(node, 'pragma'):
#             self.generic_visit(node)
#             return node

#         print('from te:', hasattr(node, 'from_tensor_expr'))
#         print(unparse(node))
        
#         # The loop can be either parallel or not parallel
#         # Parallel loops can be from either original program or expanded ops
#         # If the loop is parallel and is from expanded tensor expressions, just
#         # return the node.
#         # If the loop is parallel and is original, insert a barrier after every write.
#         # If the loop is not parallel and is from expanded tensor expressions,
#         # return the node + barrier. 
#         # If the loop is not parallel and is original, insert a barrier after every write.
#         if '#pragma parallel' in node.pragma:
#             if hasattr(node, 'from_tensor_expr'):
#                 return node
#             else:
#                 AppendBarrierToWrite().visit(node)
#                 return node            
#         else:            
#             if hasattr(node, 'from_tensor_expr'):
#                 return [node, to_ast_node('tl.debug_barrier()')]
#             else:
#                 AppendBarrierToWrite().visit(node)
#                 return node