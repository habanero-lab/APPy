import textwrap
#import ast
import ast_comments as ast
import appy
from appy.codegen.typesys import build_type_from_value, get_tl_dtype_from_str
from appy.codegen.typesys import Tensor as TensorType
from appy.codegen.typesys import Constant as ConstantType
import appy.config as config


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

        imports = textwrap.dedent(f'''
                    import {appy.config.tensorlib}
                    import numpy as np
                    import triton
                    import triton.language as tl
                    from triton.language.extra import libdevice
                    from triton.language import debug_barrier

                    def init_to_zero(name):
                        return lambda nargs: nargs[name].zero_()
                ''')
        if appy.config.tensorlib != 'torch':
            imports += 'import torch\n'

        self.module = ast.parse(imports)
        self.arg_names = [x.arg for x in self.func.args.args]

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
        from ..high_level_transforms.select_num_warps import SelectNumWarps
        from ..high_level_transforms.hoist_acc import HoistAccumulators
        from ..high_level_transforms.check_for_assign_pragma import CheckAssignPragma
        from ..high_level_transforms.insert_before_loop import InsertRangeVar
        from ..high_level_transforms.process_data_pragma import ProcessDataPragma
        from ..high_level_transforms.process_reduction_pragma import ProcessReductionPragma
        from ..high_level_transforms import add_entry_exit_data_transfer
        from ..high_level_transforms import mark_reduction_stmts
        from ..high_level_transforms import process_simd_clause
        from ..high_level_transforms import identify_readonly_scalars
        from ..high_level_transforms import process_prange

        # Perform high-level transformations
        func = self.func
        func.decorator_list = []
        #func = InsertRangeVar().visit(func)
        func = RewriteAugAssign().visit(func)
        func = RewriteRange().visit(func)
        if self.options.get('dim_info'):
            dim_info = self.options.get('dim_info')
            func = AddDimToSlice(dim_info).visit(func)

        func = add_entry_exit_data_transfer.transform(func, self.options)
        func = PragmaLinker().visit(func)    
        func = ConvertSeqLoop().visit(func)
        func = CheckAssignPragma(self.arg_val_map).visit(func)
        func = SelectNumWarps().visit(func)
        func = HoistAccumulators().visit(func)
        
        #func = InsertInitialization().visit(func) 
        func = RewriteTensorOperation(self.options, self.arg_val_map).visit(func)
        self.func = ast.fix_missing_locations(func)
        func = process_prange.transform(func)
        func = InsertRangeVar().visit(func)
        # Just run this twice for now - maybe we can do better
        func = RewriteRange().visit(func)
        func = PragmaLinker().visit(func)

        if self.options.get('no_barrier_after_write'):
            pass
        else:
            func = InsertBarrier().visit(func)
            func = RemoveBarrierInsideTE().visit(func)

        func = mark_reduction_stmts.transform(func)
        func = identify_readonly_scalars.transform(func)
        func = process_simd_clause.transform(func)
        func = ProcessReductionPragma().visit(func)
        func = ProcessDataPragma().visit(func)

        if self.options.get('dump_final_appy'): 
            print('dump final APPy code:')
            print(ast.unparse(AddBackComments().visit(func)))

        # Perform host and device code generation
        # RewritePFor rewrites the original function and also generates a triton kernel
        from .gen_host_code import RewritePFor
        launcher_func = RewritePFor(self.module, self.options, self.arg_val_map).visit(func)
        #launcher_func.decorator_list = []
        m = self.module
        m.body += [launcher_func]
        return ast.unparse(m)


class AddBackComments(ast.NodeTransformer):
    def visit_For(self, node):
        node = self.generic_visit(node)
        if hasattr(node, 'pragma'):
            return [ast.Comment(value=node.pragma, inline=False), node]
        return node