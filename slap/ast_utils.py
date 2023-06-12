import ast_comments as ast

def dump(node):
    print(ast.dump(node))

def get_func_name(tree):
    return tree.body[0].name 

def get_arg_names(func):
    args = [x.arg for x in func.args.args]
    return args