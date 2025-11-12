import ast

class AnalyzeStoredArrays(ast.NodeVisitor):
    def __init__(self):
        self.stored_arrays = set()

    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Store):
            self.stored_arrays.add(node.value.id)
        self.generic_visit(node)

class InsertDataMovement(ast.NodeTransformer):
    '''
    Insert data movement operations (e.g., host to device and device to host
    transfers) into the AST where necessary.
    '''
    def __init__(self, val_map, h2d_map):
        self.val_map = val_map
        self.h2d_map = h2d_map

    def is_numpy_array(self, val):
        return f'{type(val).__module__}.{type(val).__name__}' == 'numpy.ndarray'
    
    def is_torch_cpu_tensor(self, val):
        return f'{type(val).__module__}.{type(val).__name__}' == 'torch.Tensor' and val.device.type == 'cpu'

    def visit_FunctionDef(self, node):
        to_device_assigns = []
        for var, val in self.val_map.items():            
            if self.is_numpy_array(val) or self.is_torch_cpu_tensor(val):
                # For each `var` insert two statements if is a numpy array:
                # 1. `__tc_var = torch.from_numpy(var)`
                # 2. `__tg_var = __tc_var.to('cuda')`

                tc_assign = ast.Assign(
                    targets=[ast.Name(id=f'__tc_{var}', ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='torch', ctx=ast.Load()),
                            attr='from_numpy',
                            ctx=ast.Load()
                        ),
                        args=[ast.Name(id=var, ctx=ast.Load())],
                        keywords=[],                        
                    )
                ) if self.is_numpy_array(val) else ast.Assign(
                    targets=[ast.Name(id=f'__tc_{var}', ctx=ast.Store())],
                    value=ast.Name(id=var, ctx=ast.Load())
                )
                to_device_assigns.append(tc_assign)

                to_device_assigns.append(ast.Assign(
                    targets=[ast.Name(id=f'__tg_{var}', ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=f'__tc_{var}', ctx=ast.Load()),
                            attr='to',
                            ctx=ast.Load()
                        ),
                        args=[ast.Constant('cuda')],
                        keywords=[],                        
                    )   
                ))
                self.h2d_map[var] = f'__tg_{var}'

        store_analyzer = AnalyzeStoredArrays()
        store_analyzer.visit(node)        
        to_host_assigns = []
        for var, val in self.val_map.items():
            if var not in store_analyzer.stored_arrays:
                continue
            
            if self.is_numpy_array(val) or self.is_torch_cpu_tensor(val):
                # Insert `__torch_gpu_var.copy_(__torch_cpu_var)` after the loop
                to_host_assigns.append(
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=f'__tc_{var}', ctx=ast.Load()),
                                attr='copy_',
                                ctx=ast.Load()
                            ),
                            args=[ast.Name(id=f'__tg_{var}', ctx=ast.Load())],
                            keywords=[],                        
                        )
                    )
                )
        
        # Insert to_device_assigns at the start and to_host_assigns at the end
        node.body = to_device_assigns + node.body + to_host_assigns
        return node
    
def transform(tree, val_map):
    h2d_map = {}
    tree = InsertDataMovement(val_map, h2d_map).visit(tree)
    return tree, h2d_map