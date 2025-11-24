import ast
from ast import unparse

class MaskPropagation(ast.NodeTransformer):
    def __init__(self):
        self.var_mask = {}

    def visit_Name(self, node: ast.Name):
        if node.id in self.var_mask:
            node.mask = self.var_mask[node.id]
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
    
    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        # Get the masks of the elements
        masks = [elt.mask for elt in node.comparators if hasattr(elt, 'mask')]
        # Check if the masks are all the same
        if masks:
            if len(set(masks)) == 1:
                node.mask = masks[0]
            else:
                raise NotImplementedError("Comparisons have different masks, this is not yet supported")
        return node
    
    def visit_IfExp(self, node: ast.IfExp):
        self.generic_visit(node)
        masks = [elt.mask for elt in [node.test, node.body, node.orelse] if hasattr(elt, 'mask')]
        if masks:
            if len(set(masks)) == 1:
                node.mask = masks[0]
            else:
                raise NotImplementedError("If expressions have different masks, this is not yet supported")
        return node

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        # If the slice is not a tuple, make it a tuple
        elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        # Get the masks of the elements
        masks = [elt.mask for elt in elts if hasattr(elt, 'mask')]
        # Check if the masks are all the same
        if masks:
            if len(set(masks)) == 1:
                node.mask = masks[0]
            else:
                raise NotImplementedError("Slices have different masks, this is not yet supported")
        return node

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        # if the right hand side has attribute "mask", then the target gets its mask
        if hasattr(node.value, 'mask'):
            node.targets[0].mask = node.value.mask
            # if the target is a Name, add it to the masked_vars dict
            if isinstance(node.targets[0], ast.Name):
                self.var_mask[node.targets[0].id] = node.value.mask
        return node
    
    def visit_Call(self, node):
        self.generic_visit(node)
        # The `appy.vidx` is a special function call which has a mask (root mask)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'vidx':
            args = node.args
            node.mask = f'{unparse(args[0])} + tl.arange(0, {unparse(args[1])}) < {unparse(args[2])}'
            
        # Call `appy.where` has a special mask to attach. The first argument is a condition,
        # and if any of the other two arguments are array loads, they need to have an extra 
        # mask of the condition. If arg1 already have masks, the new mask should be condition & old_mask;
        # If arg2 already have masks, the new mask should be (~condition) & old_mask since
        # arg2 implies condition to be False, i.e. (~condition).
        elif isinstance(node.func, ast.Attribute) and node.func.attr == 'where':
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


def visit(tree):
    return MaskPropagation().visit(tree)