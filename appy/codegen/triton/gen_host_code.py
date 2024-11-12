import textwrap
from ast import unparse
import appy
from appy.ast_utils import *
from ..high_level_transforms.utils import *
from copy import deepcopy
import appy.codegen.typesys as typesys
from collections import OrderedDict


class RewritePFor(ast.NodeTransformer):
    def __init__(self, module, options, arg_val_map):
        self.module = module
        self.options = options
        self.kernel_count = 0
        self.arg_val_map = arg_val_map

    def create_new_kernel_function(self):
        kernel_name = f'_kernel{self.kernel_count}'
        self.kernel_count += 1
        k_params = self.make_kernel_parameters(self.extracted_args)
        kf = ast.parse(textwrap.dedent(f'''
            @triton.jit
            def {kernel_name}({', '.join(k_params)}):
                pass
        ''')).body[0]
        self.kf = kf
        return kf

    def make_kernel_parameters(self, arg_dim_map):
        newargs = []
        for name, (ty, ndim) in arg_dim_map.items():        
            if ty == 'const':
                newargs.append(f'{name}: tl.constexpr')
            else:
                newargs.append(name)

            if ndim > 1:
                for d in range(ndim):
                    #newargs.append(f'{name}_shape_{d}')
                    newargs.append(f'{name}_stride_{d}')        
        return newargs

    def make_kernel_actual_arguments(self, arg_dim_map):
        newargs = []
        for name, (ty, ndim) in arg_dim_map.items():
            arg = name
            if name in self.arg_val_map and appy.get_type_str(self.arg_val_map[name]) in ["numpy.float64", "numpy.float32"]:
                arg = f'float({name})'
            if appy.config.tensorlib == 'cupy' and ty == 'tensor':
                arg = f'torch.as_tensor({name}, device="cuda")'
            newargs.append(arg)
            if ndim > 1:
                for d in range(ndim):
                    if appy.config.tensorlib == 'cupy':
                        newargs.append(f'torch.as_tensor({name}, device="cuda").stride({d})')
                    else:
                        newargs.append(f'{name}.stride({d})')
        return newargs

    def make_triton_configs(self, configs, init_hook):    
        keys = []
        for name, (ty, ndim) in self.extracted_args.items():
            if ty == 'tensor':
                for d in range(ndim):
                    keys.append(f'"{name}_stride_{d}"') 
        # for param in self.make_kernel_parameters(self.extracted_args):
        #     if '_shape_' in param:
        #         keys.append('"' + param.replace(': tl.constexpr', '') + '"')
        
        triton_configs = []
        for config in configs:
            meta_args = []
            for meta_arg in ['num_warps', 'num_stages']:
                val = config.pop(meta_arg, None)
                if val:
                    meta_args.append([meta_arg, val])
            triton_config = f'triton.Config({config}'
            if init_hook:
                op, var = init_hook
                if op == 'sum':
                    triton_config += f',pre_hook=init_to_zero("{var}")' 
                else:
                    assert False            

            for k, v in meta_args:
                triton_config += f', {k}={v}'
            
            triton_config += ')'
            triton_configs.append(triton_config)
                    
        code = textwrap.dedent(f'''
        @triton.autotune(
            configs=[{','.join(triton_configs)}],
            key=[{','.join(keys)}],
        )
        ''')        
        return code


    def visit_For(self, node: ast.For):
        if hasattr(node, 'pragma'):
            from ..high_level_transforms.block_loop import BlockLoop
            from ..high_level_transforms.rewrite_call import RewriteAPPyCall
            from ..high_level_transforms.get_loaded_names import ExtractArguments

            pragma = node.pragma
            num_warps = 4
            p = get_pragma_property(pragma, 'num_warps') 
            if p:
                num_warps = int(p)

            node = BlockLoop().visit(node)
            node = RewriteAPPyCall().visit(node)            

            self.extracted_args = {}
            ExtractArguments(self.extracted_args).visit(node)
            
            # Rearrange the args to make tl.constexpr args the last
            reordered = OrderedDict()
            for name, (ty, ndim) in self.extracted_args.items():
                if ty == 'const':
                    continue
                else:
                    reordered[name] = (ty, ndim)

            for name, (ty, ndim) in self.extracted_args.items():
                if ty == 'const':
                    reordered[name] = (ty, ndim)
            self.extracted_args = reordered
            #exit(1)
                    

            from .gen_device_code import TritonKernelTransformer
            grid = []
            kernel_code = TritonKernelTransformer(grid).visit(node)                 
            kf = self.create_new_kernel_function()
            kf.body += kernel_code
            meta_grid = f'kernel_grid = lambda META: ({",".join(grid)},)'                
            #print(meta_grid)
            #print(self.options)
                              
            k_args = self.make_kernel_actual_arguments(self.extracted_args)
            
            if self.options.get('tune'):  
                configs = []
                for key, values in self.options.get('tune').items():                          
                        
                    for value in values:
                        configs.append({key: value})
                    # Remove this tuning parameter from argument list
                    # Note that DEFAULT_BLOCK may not be in the argument list
                    if key in k_args:
                        k_args.remove(key)
                    # Update the grid 
                    meta_grid = meta_grid.replace(key, f'META["{key}"]')
                init_hook = None
                if hasattr(node, 'init_hook'):
                    init_hook = node.init_hook
                tune_code = self.make_triton_configs(configs, init_hook)                
                kf = to_ast_node(tune_code + unparse(kf))

            if self.options.get('configs'):
                for config in self.options.get('configs'):
                    for key, values in config.items(): 
                        if key in k_args:
                            k_args.remove(key)
                            # Update the grid 
                            meta_grid = meta_grid.replace(key, f'META["{key}"]')
                tune_code = self.make_triton_configs(self.options.get('configs'), None)
                kf = to_ast_node(tune_code + unparse(kf))
            

            #print(unparse(kf))
            
            self.module.body.append(kf)
            
            if self.options.get('tune') or self.options.get('configs'):  
                launch_stmt = f'fn = {kf.name}[kernel_grid]({",".join(k_args)})'
            else:
                launch_stmt = f'fn = {kf.name}[kernel_grid]({",".join(k_args)}, num_warps={num_warps})'
            new_nodes = [to_ast_node(meta_grid), to_ast_node(launch_stmt) ]

            if self.options.get('print_ptx'):
                #new_nodes.append(to_ast_node('print(fn.asm["ttgir"])'))
                new_nodes.append(to_ast_node('print(fn.asm["ptx"])'))
            return new_nodes
        else:
            self.generic_visit(node)
            return node

    