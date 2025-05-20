import ast
from appy.ast_utils import *
from ast import unparse

class ScanLoop(ast.NodeTransformer):
    def __init__(self):
        self.masked_vars = {}

    def visit_Assign(self, node: ast.Assign):
        if ast.unparse(node.value).startswith('vidx('):
            dump_code(node)
            dump(node)
            args = node.value.args
            self.masked_vars[node.targets[0].id] = f'{unparse(args[0])} + tl.arange(0, {unparse(args[1])}) < {unparse(args[2])}'

        return node

class MaskPropagation(ast.NodeTransformer):
    def __init__(self, masked_vars):
        self.masked_vars = masked_vars

    def visit_Name(self, node: ast.Name):
        if node.id in self.masked_vars:
            node.mask = self.masked_vars[node.id]
        return node

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        masks = []
        for op in [node.left, node.right]:
            if hasattr(op, 'mask'):
                masks.append(op.mask)
        if len(masks) == 2:
            assert masks[0] == masks[1]

        if len(masks) >= 1:
            node.mask = masks[0]
        return node
    
    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.generic_visit(node)
        if hasattr(node.operand, 'mask'):
            node.mask = node.operand.mask
        return node

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        # If the slice is not a tuple, make it a tuple
        elts = []
        if not isinstance(node.slice, ast.Tuple):
            elts = [node.slice]
        else:
            elts = node.slice.elts
        
        mask = None
        # If one of the slices has attribute "mask", then the whole expression is masked
        for elt in elts:
            if hasattr(elt, 'mask'):
                if mask is None:
                    mask = elt.mask
                else:
                    if mask != elt.mask:
                        raise NotImplementedError("Slices have different masks, this is not yet supported")
        if mask is not None:
            node.mask = mask
        return node

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        # if the right hand side has attribute "mask", then the target gets its mask
        if hasattr(node.value, 'mask'):
            node.targets[0].mask = node.value.mask
            # if the target is a Name, add it to the masked_vars dict
            if isinstance(node.targets[0], ast.Name):
                self.masked_vars[node.targets[0].id] = node.value.mask
        return node
    
    def visit_Call(self, node):
        self.generic_visit(node)
        # Call `appy.where` has a special mask to attach. The first argument is a condition,
        # and if any of the other two arguments are array loads, they need to have an extra 
        # mask of the condition. If arg1 already have masks, the new mask should be condition & old_mask;
        # If arg2 already have masks, the new mask should be (~condition) & old_mask since
        # arg2 implies condition to be False, i.e. (~condition).
        if unparse(node).startswith('appy.where('):
            condition = node.args[0]
            arg1 = node.args[1]
            arg2 = node.args[2]
            if isinstance(arg1, ast.Subscript):
                if hasattr(arg1, 'mask'):
                    arg1.mask = f'({unparse(condition)}) & ({arg1.mask})'
                else:
                    arg1.mask = f'({unparse(condition)})'
            if isinstance(arg2, ast.Subscript):
                if hasattr(arg2, 'mask'):
                    arg2.mask = f'(~{unparse(condition)}) & ({arg2.mask})'
                else:
                    arg2.mask = f'(~{unparse(condition)})'
        return node


class AttachMaskInfo(ast.NodeTransformer):
    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        masked_vars = {}
        for child in node.body:
            if isinstance(child, ast.Assign):
                if ast.unparse(child.value).startswith('vidx('):
                    args = child.value.args
                    target = child.targets[0].id
                    masked_vars[target] = f'{unparse(args[0])} + tl.arange(0, {unparse(args[1])}) < {unparse(args[2])}'
        visitor = MaskPropagation(masked_vars)
        visitor.visit(node)
        return node

