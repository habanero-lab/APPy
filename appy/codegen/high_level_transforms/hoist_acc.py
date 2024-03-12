from ast import unparse
from appy.ast_utils import *
from .utils import *

class InspectAssign(ast.NodeVisitor):
    def __init__(self) -> None:
        self.candidates = {}

    def visit_Assign(self, node):
        if hasattr(node, 'reduce') and node.reduce == '+':
            if isinstance(node.targets[0], ast.Subscript):
                target = node.targets[0]
                if target not in self.candidates:
                    self.candidates[target] = None
                    if hasattr(node, 'pragma'):
                        self.candidates[target] = node.pragma
           

class Substitute(ast.NodeTransformer):
    def __init__(self, targets):
        self.targets = targets

    def visit_Subscript(self, node):
        for sub, var in self.targets.items():
            if unparse(node) == unparse(sub):
                return var
        return node


class CheckLoopVariant(ast.NodeVisitor):
    def __init__(self, loop_var):
        self.loop_var = loop_var
        self.is_inv = True

    def visit_Name(self, node):
        if node.id == self.loop_var.id:
            self.is_inv = False


class HoistAccumulators(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def get_new_var_name(self):
        name = f'__hoist_acc_{self.count}'
        self.count += 1
        return name

    def visit_For(self, node: ast.For):
        node = self.generic_visit(node)
        #print('dump for loop')
        #print(unparse(node))
    
        visitor = InspectAssign()
        visitor.visit(node) 

        target_map = {}
        new_assigns_before = []
        new_assigns_after = []
        index_var = node.target
        for sub, pragma in visitor.candidates.items():
            checker = CheckLoopVariant(index_var)
            checker.visit(sub)
            
            # Skip if not a loop invariant
            if not checker.is_inv:
                continue

            # Only the "le" subscripts can be hoisted (stored in registers)
            if pragma:
                if pragma.startswith('#pragma atomic'):
                    continue

                if '=>' in pragma:
                    slice_map = parse_pragma(pragma)
                    if len(slice_map) != 1:
                        continue

                    if not list(slice_map.values())[0]['single_block']:
                        continue
            
            var = new_name_node(self.get_new_var_name())
            target_map[sub] = var
            assign_before = to_ast_node(f'{unparse(var)} = {unparse(sub)}')
            assign_after = to_ast_node(f'{unparse(sub)} = {unparse(var)}')
            if pragma:
                assign_before.pragma = assign_after.pragma = pragma
            new_assigns_before.append(assign_before)
            new_assigns_after.append(assign_after)
    
            substitutor = Substitute(target_map)
            substitutor.visit(node)

        if len(new_assigns_before) > 0:
            node = new_assigns_before + [node] + new_assigns_after        
        return node