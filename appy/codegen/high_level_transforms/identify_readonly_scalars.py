import ast
from .get_loaded_names import ExtractArguments
from .process_reduction_pragma import ReplaceNameWithSubscript

class IdentifySharedVars(ast.NodeTransformer):
    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        if hasattr(node, 'pragma_dict') and 'parallel_for' in node.pragma_dict:
            name_visitor = ExtractArguments()
            name_visitor.visit(node)
            range_names = []
            # Extract the range names from the arguments of the for loop
            # For example, if the loop is "for xx in range(M, N)", then range_names = ['M', 'N']
            for arg in node.iter.args:
                if isinstance(arg, ast.Name):
                    range_names.append(arg.id)
            
            # Get the scalar names that are read but not written in the for loop
            readonly_scalars = []
            for var, type_and_ndim in name_visitor.read_names.items():
                if type_and_ndim[0] == 'scalar' and var not in name_visitor.write_names:
                    readonly_scalars.append(var)
            
            # Exclude the range names from the shared scalars
            readonly_scalars = [s for s in readonly_scalars if s not in range_names]

            # Update the pragma dictionary
            if readonly_scalars:
                node.pragma_dict['to'] = node.pragma_dict.get('to', []) + readonly_scalars
                # Rewrite a scalar reference to a subscript with slice 0
                node = ReplaceNameWithSubscript(readonly_scalars).visit(node)

            # Add tensor variables to "to" and "from" clause
            for var, type_and_ndim in name_visitor.read_names.items():
                if type_and_ndim[0] == 'tensor':
                    node.pragma_dict['to'] = node.pragma_dict.get('to', []) + [var]
                    node.pragma_dict['from'] = node.pragma_dict.get('from', []) + [var]
        return node


def transform(node):
    visitor = IdentifySharedVars()
    node = visitor.visit(node)
    return node