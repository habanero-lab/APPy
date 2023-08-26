import ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy

class AppendBarrierToWrite(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign):        
        if isinstance(node.targets[0], ast.Subscript):
            #print('to insert after write')
            #dump(node)
            return node, to_ast_node('tl.debug_barrier()')
        else:
            return node

class InsertBarrier(ast.NodeTransformer):
    def visit_For(self, node: ast.For):
        if not hasattr(node, 'pragma'):
            return node
        
        # The loop can be either parallel or not parallel
        # Parallel loops can be from either original program or expanded ops
        # If the loop is parallel and is from expanded tensor expressions, just
        # return the node.
        # If the loop is parallel and is original, insert a barrier after every write.
        # If the loop is not parallel and is from expanded tensor expressions,
        # return the node + barrier. 
        # If the loop is not parallel and is original, insert a barrier after every write.
        if '#pragma parallel' in node.pragma:
            if hasattr(node, 'from_tensor_expr'):
                return node
            else:
                AppendBarrierToWrite().visit(node)
                return node            
        else:            
            if hasattr(node, 'from_tensor_expr'):
                return [node, to_ast_node('tl.debug_barrier()')]
            else:
                AppendBarrierToWrite().visit(node)
                return node