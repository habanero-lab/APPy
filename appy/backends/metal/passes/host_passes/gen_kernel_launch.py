import ast
import textwrap as tw

class GenKernelLaunch(ast.NodeTransformer):
    def __init__(self, loop_name, val_map):
        self.loop_name = loop_name
        self.val_map = val_map
        self.replaced_loop = None

    def visit_For(self, node):
        assert len(node.iter.args) == 1
        num_iters = ast.unparse(node.iter.args[0])

        args = [num_iters]
        for k, v in self.val_map.items():
            if k == num_iters:
                continue

            if type(v) == int:
                args.append(f"np.int32({k})")
            elif type(v) == float:
                args.append(f"np.float32({k})")
            elif type(v) == bool:
                args.append(f"np.bool({k})")
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
    

def transform(node, loop_name, val_map):
    transformer = GenKernelLaunch(loop_name, val_map)
    return transformer.visit(node), transformer.replaced_loop