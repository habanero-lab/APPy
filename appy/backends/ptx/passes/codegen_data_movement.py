import ast

class InsertDataMovement(ast.NodeTransformer):
    '''
    Insert data movement operations (e.g., host to device and device to host
    transfers) into the AST where necessary.
    '''
    def __init__(self, val_map):
        self.val_map = val_map

    def visit_Module(self, node):
        to_device_assigns = []
        for var, val in self.val_map.items():
            ty = type(val)
            if f'{ty.__module__}.{ty.__name__}' == 'numpy.ndarray':
                # Insert `var_gpu = gpuarray.to_gpu(var)` before the loop
                to_gpu_call = ast.Assign(
                    targets=[ast.Name(id=f'{var}_gpu', ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='gpuarray', ctx=ast.Load()),
                            attr='to_gpu',
                            ctx=ast.Load()
                        ),
                        args=[ast.Name(id=var, ctx=ast.Load())],
                        keywords=[],                        
                    )
                )
                ast.fix_missing_locations(to_gpu_call)
                to_device_assigns.append(to_gpu_call)

        to_host_assigns = []
        for var, val in self.val_map.items():
            ty = type(val)
            if f'{ty.__module__}.{ty.__name__}' == 'numpy.ndarray':
                # Insert `var_gpu.get(var)` after the loop
                to_host_call = ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=f'{var}_gpu', ctx=ast.Load()),
                            attr='get',
                            ctx=ast.Load()
                        ),
                        args=[ast.Name(id=var, ctx=ast.Load())],
                        keywords=[],                        
                    )
                )
                ast.fix_missing_locations(to_host_call)
                to_host_assigns.append(to_host_call)
        
        # Insert to_device_assigns at the start and to_host_assigns at the end
        node.body = to_device_assigns + node.body + to_host_assigns
        return node