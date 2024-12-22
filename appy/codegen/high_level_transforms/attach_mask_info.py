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

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        # if one of the slices has attribute "mask", then the whole expression is masked
        # more than one slice having a mask is not yet supported
        if hasattr(node.slice, 'mask'):
            node.mask = node.slice.mask
            # print a message about this subscript having a mask now
            # print(f'{unparse(node)} has mask: {node.mask}')
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


class AttachMaskInfo(ast.NodeTransformer):
    def __init__(self):
        self.masked_vars = {}

    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        for child in node.body:
            if isinstance(child, ast.Assign):
                if ast.unparse(child.value).startswith('vidx('):
                    args = child.value.args
                    target = child.targets[0].id
                    self.masked_vars[target] = f'{unparse(args[0])} + tl.arange(0, {unparse(args[1])}) < {unparse(args[2])}'
        visitor = MaskPropagation(self.masked_vars)
        visitor.visit(node)
        return node

