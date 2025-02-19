import ast
from ast import unparse
from appy.ast_utils import *
from copy import deepcopy


class CollectTensorAccessInfo(ast.NodeVisitor):
    def __init__(self):
        self.read_tensors = {}
        self.write_tensors = {}

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name):
            if isinstance(node.ctx, ast.Load):
                if node.value.id not in self.read_tensors:
                    self.read_tensors[node.value.id] = set()
                self.read_tensors[node.value.id].add(ast.unparse(node.slice))
            elif isinstance(node.ctx, ast.Store):
                if node.value.id not in self.write_tensors:
                    self.write_tensors[node.value.id] = set()
                self.write_tensors[node.value.id].add(ast.unparse(node.slice))


class AddBarrierToMemAccess(ast.NodeTransformer):
    def __init__(self, elide_tensors):
        self.elide_tensors = elide_tensors

    def visit_Assign(self, node: ast.Assign):        
        target = node.targets[0]
        if isinstance(target, ast.Subscript):
            if target.value.id in self.elide_tensors:
                #print(f'[DEBUG] tensor {target.value.id} is elided')
                return node
            elif not hasattr(node, 'no_sync'):
                #print(f'[DEBUG] node is marked as no_sync')
                return node
            else:
                return to_ast_node('tl.debug_barrier()'), node, to_ast_node('tl.debug_barrier()')
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
            visitor = CollectTensorAccessInfo()
            visitor.visit(node)
            elide_tensors = set()
            for t in visitor.write_tensors:
                write_indices = visitor.write_tensors[t]
                # If the tensor is only written and never read, elide it
                if t not in visitor.read_tensors:
                    elide_tensors.add(t)
                else:
                    # If the tensor is only written with one index and only 
                    # read with one index, and the indices are the same, elide it
                    read_indices = visitor.read_tensors[t]
                    if len(write_indices) == 1 and len(read_indices) == 1:
                        if write_indices == read_indices:
                            elide_tensors.add(t)
            AddBarrierToMemAccess(elide_tensors).visit(node)
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
#                 AddBarrierToMemAccess().visit(node)
#                 return node            
#         else:            
#             if hasattr(node, 'from_tensor_expr'):
#                 return [node, to_ast_node('tl.debug_barrier()')]
#             else:
#                 AddBarrierToMemAccess().visit(node)
#                 return node