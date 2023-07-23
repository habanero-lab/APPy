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

def get_keyword_args(func):
    d = {}
    for keyword in func.keywords:
        d[keyword.arg] = keyword.value
    return d

def get_arg_str(func, idx):
    return ast.unparse(func.args[idx])

def is_call(node, names=None):
    if names == None:
        return isinstance(node, ast.Call)
    else:
        if type(names) not in [list, tuple]:
            names = [names]
        return isinstance(node, ast.Call) and node.func.id in names

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

def to_ast_expr(s):
    n = ast.parse(s).body[0]
    assert isinstance(n, ast.Expr)
    return n.value

def new_call_node(func_name, args):
    node = ast.Call(func=ast.Name(func_name, ctx=ast.Load()), args=args, keywords=[])
    return node

def new_name_node(name, ctx=None):
    ctx = ast.Load() if ctx == None else ctx
    return ast.Name(id=name, ctx=ctx)

def new_add_node(a, b):
    return ast.BinOp(left=a, op=ast.Add(), right=b)

def new_sub_node(a, b):
    return ast.BinOp(left=a, op=ast.Sub(), right=b)