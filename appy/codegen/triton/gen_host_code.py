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
            if appy.config.tensorlib == 'cupy' and ty == 'tensor':
                arg = f'torch.as_tensor({name}, device="cuda")'
            newargs.append(arg)
            if ndim > 1:
                for d in range(ndim):
                    if appy.config.tensorlib == 'cupy':
                        newargs.append(f'{name}.strides[{d}] / {name}.itemsize')
                    else:
                        newargs.append(f'{name}.stride({d})')
        return newargs

    def visit_For(self, node: ast.For):
        if hasattr(node, 'pragma'):
            from ..high_level_transforms.block_loop import BlockLoop
            from ..high_level_transforms.attach_mask_info import AttachMaskInfo
            from ..high_level_transforms.rewrite_call import RewriteAPPyCall
            from ..high_level_transforms import get_loaded_names
            from ..high_level_transforms import ternary_to_where

            pragma = node.pragma
            num_warps = 4
            p = get_pragma_property(pragma, 'num_warps') 
            if p:
                num_warps = int(p)

            node = BlockLoop().visit(node)
            node = ternary_to_where.transform(node)
            node = AttachMaskInfo().visit(node)
            node = RewriteAPPyCall().visit(node)            
            node, self.extracted_args = get_loaded_names.transform(node)
            
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
            self.module.body.append(kf)
            meta_grid = f'kernel_grid = lambda META: ({",".join(grid)},)'
            k_args = self.make_kernel_actual_arguments(self.extracted_args)            
            launch_stmt = f'fn = {kf.name}[kernel_grid]({",".join(k_args)}, num_warps={num_warps})'
            launch_stmt += '\ntorch.cuda.synchronize()'
            new_nodes = [to_ast_node(meta_grid), to_ast_node(launch_stmt) ]

            if self.options.get('print_ptx'):
                #new_nodes.append(to_ast_node('print(fn.asm["ttgir"])'))
                new_nodes.append(to_ast_node('print(fn.asm["ptx"])'))
            return new_nodes
        else:
            self.generic_visit(node)
            return node

    