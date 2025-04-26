import ast
from .get_loaded_names import ExtractArguments

class IdentifySharedVars(ast.NodeVisitor):
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
            shared_scalars = []
            for var, type_and_ndim in name_visitor.read_names.items():
                if type_and_ndim[0] == 'scalar' and var not in name_visitor.write_names:
                    shared_scalars.append(var)
            
            # Exclude the range names from the shared scalars
            shared_scalars = [s for s in shared_scalars if s not in range_names]

            # Update the pragma dictionary
            if 'to' in node.pragma_dict:
                node.pragma_dict['to'] += shared_scalars
            else:
                node.pragma_dict['to'] = shared_scalars


def transform(node):
    visitor = IdentifySharedVars()
    visitor.visit(node)
    return node