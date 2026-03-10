import ast
import math


def transform(tree, val_map):
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Tuple):
            val = val_map[node.value.id]
            assert len(val.shape) == len(node.slice.elts), \
                f"Dimension mismatch: indexing {node.value.id} with {ast.unparse(node.slice)} but shape is {val.shape}"

            strides = [ast.Constant(value=math.prod(val.shape[i+1:]))
                       for i in range(len(val.shape))]

            new_index = None
            for x, y in zip(node.slice.elts, strides):
                term = ast.BinOp(op=ast.Mult(), left=x, right=y)
                new_index = term if new_index is None else ast.BinOp(op=ast.Add(), left=new_index, right=term)

            node.slice = new_index
    return tree
