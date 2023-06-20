import ast_comments as ast

def dump(node):
    print(ast.dump(node))

def dump_code(node):
    print(ast.unparse(node))

def get_func_name(tree):
    return tree.body[0].name 

def get_arg_names(func):
    args = [x.arg for x in func.args.args]
    return args

def get_first_noncomment_child(node):
    for c in node.body:
        if type(c) == ast.Comment:
            continue
        return c

def to_ast_node(s):
    n = ast.parse(s).body[0]
    # if isinstance(n, ast.Expr):
    #     n = n.value
    return n