import ast
from .process_reduction_pragma import ReplaceNameWithSubscript
from . import var_access_analysis

class IdentifySharedVars(ast.NodeTransformer):
    def __init__(self, options):
        self.options = options
        self.auto_transfer = options.get('auto_transfer', True)
        if 'entry_to_device' in self.options or 'exit_to_host' in self.options:
            self.auto_transfer = False
        
    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        if hasattr(node, 'pragma_dict') and 'parallel_for' in node.pragma_dict:
            var_info = var_access_analysis.visit(node)
        
            range_names = []
            # Extract the range names from the arguments of the for loop
            # For example, if the loop is "for xx in range(M, N)", then range_names = ['M', 'N']
            for arg in node.iter.args:
                if isinstance(arg, ast.Name):
                    range_names.append(arg.id)
            
            # Get the scalar names that are read but not written in the for loop
            readonly_scalars = []
            for var, props in var_info.items():
                if props[0] == set(['scalar']) and props[2] == set(['load']):
                    readonly_scalars.append(var)
            
            # Exclude the range names from the shared scalars
            readonly_scalars = [s for s in readonly_scalars if s not in range_names]

            # Update the pragma dictionary
            if readonly_scalars:
                node.pragma_dict['to'] = node.pragma_dict.get('to', []) + readonly_scalars
                # Rewrite a scalar reference to a subscript with slice 0
                node = ReplaceNameWithSubscript(readonly_scalars).visit(node)

            # Add tensor variables to "to" and "from" clause if auto_transfer is enabled
            if self.auto_transfer:
                for var, props in var_info.items():
                    if props[0] == set(['tensor']):
                        node.pragma_dict['to'] = node.pragma_dict.get('to', []) + [var]
                        if 'store' in props[2]:
                            node.pragma_dict['from'] = node.pragma_dict.get('from', []) + [var]
        return node


def transform(node, options):
    visitor = IdentifySharedVars(options)
    node = visitor.visit(node)
    return node