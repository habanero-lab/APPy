import ast
import ast_comments as astc

def check_top_level_loop_count(m):
    top_level_for_loop = 0
    for node in m.body:
        if isinstance(node, ast.For):
            top_level_for_loop += 1
    if top_level_for_loop > 1:
        raise Exception(f"Input AST should only contain a single for loop while {top_level_for_loop} found.")
    
def check_has_pragma(m):
    for node in m.body:
        if isinstance(node, astc.Comment) and node.value.startswith('#pragma '):
            raise NotImplementedError("Pragma is not supported yet. Found pragma: " + node.value)
        
def visit(node, verbose=False):
    check_top_level_loop_count(node)
    if verbose:
        print("[Sanity Check] Loop count check has passed.")

    check_has_pragma(node)
    if verbose:
        print("[Sanity Check] Pragma check has passed.")