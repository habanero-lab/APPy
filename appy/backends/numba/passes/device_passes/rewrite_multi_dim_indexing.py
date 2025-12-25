import ast
import math

def transform(tree, val_map):
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Tuple):
            
            val = val_map[node.value.id]
            assert len(val.shape) == len(node.slice.elts), f"Dimension mismatch: indexing array {node.value.id} with index {ast.unparse(node.slice)} but has shape {val.shape}"

            strides = []
            for i in range(len(val.shape)):
                stride_i = math.prod(val.shape[i+1:])
                strides.append(ast.Constant(value=stride_i))

            new_index = None
            for x, y in zip(node.slice.elts, strides):
                if new_index is None:
                    new_index = ast.BinOp(
                        op=ast.Mult(),
                        left=x,
                        right=y
                    )
                else:
                    new_index = ast.BinOp(
                        op=ast.Add(),
                        left=new_index,
                        right=ast.BinOp(
                            op=ast.Mult(),
                            left=x,
                            right=y
                        )
                    )

            node.slice = new_index

    return tree
