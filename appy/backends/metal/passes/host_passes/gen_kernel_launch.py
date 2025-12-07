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
       
        first_array_arg = None
        for var, val in self.val_map.items():
            if hasattr(val, "dev"):
                first_array_arg = var
                break

        args = [f"{k}.buf" if hasattr(v, "buf") else k for k,v in self.val_map.items()]
        args = [x for x in args if x != num_iters]
        args.insert(0, num_iters)

        code_str = f'''
            if not hasattr({self.loop_name}, "kernel"):
                {self.loop_name}.kernel = {first_array_arg}.dev.kernel(kernel_str).function("_{self.loop_name}")
            handle = {self.loop_name}.kernel({", ".join(args)})
            del handle
        '''
        self.replaced_loop = node
        node = ast.parse(tw.dedent(code_str)).body
        return node
    

def transform(node, loop_name, val_map):
    transformer = GenKernelLaunch(loop_name, val_map)
    return transformer.visit(node), transformer.replaced_loop