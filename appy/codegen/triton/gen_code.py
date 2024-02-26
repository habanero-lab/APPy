import textwrap
import numpy as np
from copy import deepcopy
import ast
from ast_comments import unparse
from appy.codegen.typesys import build_type_from_value, get_tl_dtype_from_str
from appy.codegen.typesys import Tensor as TensorType
from appy.codegen.typesys import Constant as ConstantType
from appy.ast_utils import *

class RewriteForRange(ast.NodeTransformer):
    def __init__(self) -> None:
        self.var_count = 0

    def get_new_var(self):
        new_var = f'__rewrite_for_range_var{self.var_count}'
        self.var_count += 1
        return new_var
        
    def visit_For(self, node: ast.For):
        new_stmts = []
        if isinstance(node.iter, ast.Call) and node.iter.func.id == 'range':
            new_args = []
            for arg in node.iter.args:
                if not isinstance(arg, (ast.Name, ast.Constant)):
                    new_var = self.get_new_var()
                    new_args.append(new_name_node(new_var))                
                    new_stmts.append(
                        new_assign_node(
                            new_name_node(new_var, ctx=ast.Store()),
                            arg,
                            lineno=node.lineno
                        )
                    )                
                else:
                    new_args.append(arg)
            node.iter.args = new_args
        return new_stmts, node


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
        from ..high_level_transforms.range_rewriter import RewriteRange
        from ..high_level_transforms.rewrite_call import RewriteAPPyCall
        from ..high_level_transforms.link_pragma import PragmaLinker
        from ..high_level_transforms.aug_assign_rewriter import RewriteAugAssign
        from ..high_level_transforms.transform_tensor_pragma import RewriteTensorOperation
        from ..high_level_transforms.add_dim_to_slice import AddDimToSlice
        from ..high_level_transforms.insert_barrier import InsertBarrier, RemoveBarrierInsideTE
        #from ..high_level_transforms.insert_initialization import InsertInitialization
        from ..high_level_transforms.convert_pragma_seq_for import ConvertSeqLoop

        func = self.func
        func.decorator_list = []
        func = RewriteForRange().visit(func)
        func = RewriteAugAssign().visit(func)
        func = RewriteRange().visit(func)        
        if self.options.get('dim_info'):
            dim_info = self.options.get('dim_info')
            func = AddDimToSlice(dim_info).visit(func)

        func = PragmaLinker().visit(func)    
        func = ConvertSeqLoop().visit(func)
        
        #func = InsertInitialization().visit(func) 
        func = RewriteTensorOperation(self.options, self.arg_val_map).visit(func)
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
        from .gen_host_code import RewritePFor
        launcher_func = RewritePFor(self.module, self.options, self.arg_val_map).visit(func)
        #launcher_func.decorator_list = []
        m = self.module
        m.body += [launcher_func]
        return ast.unparse(m)
