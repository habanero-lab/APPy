import os
import ast
import textwrap as tw
from pathlib import Path
import ast_transforms as at
from ..base import Backend
from . import passes as passes

class PTXBackend(Backend):
    def __init__(self, val_map=None):
        self.val_map = val_map or {}        
        
    def codegen(self, tree, metadata):
        tree = passes.remove_appy(tree)
        tree = passes.block_loop(tree)
        tree = passes.to_unit_stmts_form(tree)
        tree, type_map = passes.attach_types(tree, self.val_map)
        tree = passes.to_pseudo_ptx(tree, self.val_map, type_map)

        #tree = passes.add_builtin_imports(tree)
        
        tree = passes.codegen_data_movement(tree, self.val_map)
        tree = passes.codegen_load_kernel(tree)
        tree = passes.codegen_pycuda_imports(tree)
        tree = passes.codegen_kernel_launch(tree)
        ast.fix_missing_locations(tree)
        return tree
        