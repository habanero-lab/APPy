import ast

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
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names

def is_add(node):
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)

def is_attr_call(node, name=None):
    if name == None:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    else:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == name

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

def to_ast_nodes(s):
    n = ast.parse(s).body
    return n

def to_ast_expr(s):
    n = ast.parse(s).body[0]
    assert isinstance(n, ast.Expr)
    return n.value

def new_call_node(func_name, args, keywords=None):
    kws = []
    if keywords:
        for k, v in keywords.items():
            kws.append(ast.keyword(arg=k, value=v))
    node = ast.Call(func=ast.Name(func_name, ctx=ast.Load()), args=args, keywords=kws)
    return node

def new_attr_node(value, attr):
    node = ast.Attribute(value=value, attr=attr, ctx=ast.Load())
    return node

def new_attr_call_node(func_name, args, keywords=None):
    m, f = func_name.split('.')
    kws = []
    if keywords:
        for k, v in keywords.items():            
            kws.append(ast.keyword(arg=k, value=v))
    node = ast.Call(func=ast.Attribute(value=ast.Name(id=m, ctx=ast.Load()), attr=f, ctx=ast.Load()), \
        args=args, keywords=kws)
    return node

def new_name_node(name, ctx=None):
    ctx = ast.Load() if ctx == None else ctx
    return ast.Name(id=name, ctx=ctx)

def new_const_node(val):
    return ast.Constant(value=val)

def new_assign_node(target, value, lineno=None):
    if lineno:
        return ast.Assign(targets=[target], value=value, lineno=lineno)
    else:
        return ast.Assign(targets=[target], value=value)

def new_add_node(a, b):
    return ast.BinOp(left=a, op=ast.Add(), right=b)

def new_mul_node(a, b):
    return ast.BinOp(left=a, op=ast.Mult(), right=b)

def new_sub_node(a, b):
    return ast.BinOp(left=a, op=ast.Sub(), right=b)

def new_for_loop(target, low, up, step):
    loop = ast.For(target=target, iter=new_call_node('range', [low, up, step]), body=[], \
        orelse=[], type_ignores=[])
    return loop

def append_new_argument(f: ast.FunctionDef, arg_name: str, annotation=None):
    class ArgumentAppender(ast.NodeTransformer):
        def __init__(self, arg_name, annotation):
            self.arg_name = arg_name
            self.annotation = annotation

        def visit_FunctionDef(self, node):
            new_arg = ast.arg(arg=self.arg_name, annotation=ast.parse(self.annotation).body[0].value if self.annotation else None)
            node.args.args.append(new_arg)
            return node

    appender = ArgumentAppender(arg_name, annotation)
    new_function = appender.visit(f)
    return new_function

def is_array_access(node):
    return isinstance(node, ast.Subscript)

def get_array_name(node: ast.Subscript):
    assert is_array_access(node)
    return node.value.id
