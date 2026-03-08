import ast
import textwrap as tw
from ..constants import SIMD_WIDTH

class GenKernelLaunch(ast.NodeTransformer):
    def __init__(self, loop_name, val_map, use_simd):
        self.loop_name = loop_name
        self.val_map = val_map
        self.use_simd = use_simd
        self.replaced_loop = None

    def visit_For(self, node):
        range_args = node.iter.args
        assert len(range_args) in (1, 2), \
            f"The parallel for-range loop must have 1 or 2 arguments, got: {ast.unparse(node.iter)}"

        if len(range_args) == 1:
            low, up = "0", ast.unparse(range_args[0])
        else:
            low, up = ast.unparse(range_args[0]), ast.unparse(range_args[1])

        n_iters = up if low == "0" else f"({up}) - ({low})"
        num_iters = f"({n_iters}) * {SIMD_WIDTH}" if self.use_simd else n_iters

        args = [num_iters]
        for k, v in self.val_map.items():
            if type(v) == int:
                args.append(f"np.int32({k})")
            elif type(v) == float:
                args.append(f"np.float32({k})")
            elif type(v) == bool:
                args.append(f"np.bool_({k})")
            else:
                args.append(k)

        #print(args)

        code_str = f'''
            if not hasattr({self.loop_name}, "kernel"):
                {self.loop_name}.kernel = device.kernel(kernel_str).function("_{self.loop_name}")
            handle = {self.loop_name}.kernel({", ".join(args)})
            del handle
        '''
        self.replaced_loop = node
        node = ast.parse(tw.dedent(code_str)).body
        return node
    

def transform(node, loop_name, val_map, use_simd=False):
    transformer = GenKernelLaunch(loop_name, val_map, use_simd)
    return transformer.visit(node), transformer.replaced_loop