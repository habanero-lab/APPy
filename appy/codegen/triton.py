import os
import re
import torch
import textwrap
import numpy as np
from copy import deepcopy
import ast_comments as ast
from ast_comments import unparse
from .typesys import build_type_from_value, get_tl_dtype_from_str
from .typesys import Tensor as TensorType
from .typesys import Constant as ConstantType
from ..ast_utils import *

class TritonBackend(object):
    def __init__(self, ast_tree, arg_values, **options):
        self.options = options
        for node in ast_tree.body:
            if isinstance(node, ast.FunctionDef):
                self.func = node
                break
        self.arg_values = list(arg_values)
        for keyword_arg in self.func.args.defaults:
            assert isinstance(keyword_arg, ast.Constant)
            self.arg_values.append(keyword_arg.value)
        
        self.module = ast.parse(textwrap.dedent('''
            import numpy as np
            import torch
            import triton
            import triton.language as tl
            from triton.language import debug_barrier
            import appy
            from appy import vidx, vindex

            def init_to_zero(name):
                return lambda nargs: nargs[name].zero_()
        '''
        ))
        self.arg_names = get_arg_names(self.func)

        self.arg_val_map = {}
        for name, val in zip(self.arg_names, self.arg_values):
            self.arg_val_map[name] = val
       
        #self.arg_types = [build_type_from_value(x) for x in arg_values]
        
        # self.arg_type_map = {}
        # for name, type in zip(self.arg_names, self.arg_types):
        #     self.arg_type_map[name] = type
        
        # print('type defined now')
        self.kernel_count = 0
        self.var_count = 0

    def codegen(self):
        from .high_level_transforms.range_rewriter import RewriteRange
        from .high_level_transforms.rewrite_call import RenameTorchToTriton
        from .high_level_transforms.link_pragma import PragmaLinker
        from .high_level_transforms.aug_assign_rewriter import RewriteAugAssign
        from .high_level_transforms.transform_tensor_pragma import RewriteTensorOperation
        from .high_level_transforms.add_dim_to_slice import AddDimToSlice
        from .high_level_transforms.insert_barrier import InsertBarrier, RemoveBarrierInsideTE
        from .high_level_transforms.insert_initialization import InsertInitialization
        from .high_level_transforms.convert_pragma_seq_for import ConvertSeqLoop

        func = self.func
        func.decorator_list = []
        func = RewriteAugAssign().visit(func)
        func = RewriteRange().visit(func)        
        if self.options.get('dim_info'):
            dim_info = self.options.get('dim_info')
            func = AddDimToSlice(dim_info).visit(func)

        #print(unparse(func))
        #exit(1)
        func = PragmaLinker().visit(func)    
        func = ConvertSeqLoop().visit(func)
        
        #func = InsertInitialization().visit(func) 
        func = RewriteTensorOperation(self.options, self.arg_val_map).visit(func)
        #func = RenameTorchToTriton().visit(func)
        self.func = ast.fix_missing_locations(func)
        

        func = PragmaLinker().visit(func)
        if self.options.get('no_barrier_after_write'):
            pass
        else:
            func = InsertBarrier().visit(func)
            func = RemoveBarrierInsideTE().visit(func)

        if self.options.get('dump_final_appy'): 
            print('dump final APPy code:')            
            print(ast.unparse(func))

        #exit(1)
        from .rewrite_pfor import RewritePFor
        launcher_func = RewritePFor(self.module, self.options, self.arg_val_map).visit(func)
        #launcher_func.decorator_list = []
    
        # lf = ast.FunctionDef(name='kernel', args=self.func.args, body=[], decorator_list=[], lineno=self.func.lineno)
        # self.lf = lf        

        # for node in self.func.body:
        #     if isinstance(node, ast.For) and hasattr(node, 'pragma'):
        #         pragma = node.pragma
        #         p = self.get_pragma_property(pragma, 'num_warps')
        #         num_warps = 4
        #         if p:
        #             num_warps = int(p)
                
        #         kf = self.create_new_kernel_function()
                
        #         self.allBlockDims = ['x', 'y', 'z']
        #         self.usedBlockDims = []

        #         #self.gen_parallel_for(node, pragma)
        #         from .triton_transformer import TritonKernelTransformer
        #         grid = []
        #         kernel_code = TritonKernelTransformer(grid).visit(node)                 
        #         kf.body += kernel_code
        #         meta_grid = f'kernel_grid = lambda META: ({",".join(grid)},)'                
        #         #print(meta_grid)
        #         #print(self.options)
                
        #         #exit(2)                
                
        #         k_args = self.get_kernel_function_arguments()
        #         if self.options.get('tune'):                    
        #             configs = []
        #             for key, values in self.options.get('tune').items():
        #                 if key == 'APPY_BLOCK':
        #                     append_new_argument(kf, key, annotation='tl.constexpr')                            
                            
        #                 for value in values:
        #                     configs.append({key: value})
        #                 # Remove this tuning parameter from argument list
        #                 # Note that DEFAULT_BLOCK may not be in the argument list
        #                 if key in k_args:
        #                     k_args.remove(key)
        #                 # Update the grid 
        #                 meta_grid = meta_grid.replace(key, f'META["{key}"]')
        #             tune_code = self.make_triton_configs(configs)
        #             kf = to_ast_node(tune_code + unparse(kf))
        #             #print(unparse(kf))
                    
        #         self.append_stmts(self.lf, meta_grid)
        #         self.module.body.append(kf)
        #         self.append_stmts(self.lf, f'fn = {kf.name}[kernel_grid]({",".join(k_args)})')
        #         if 'print_ptx' in self.options and self.options['print_ptx']:
        #             self.append_stmts(self.lf, 'print(fn.asm["ttgir"])')
            
        #         #self.append_stmts(self.lf, 'exit(1)')
        #     else:
        #         self.append_node(self.lf, node)
            
            
        m = self.module
        m.body += [launcher_func]
        return ast.unparse(m)
