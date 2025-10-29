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
        # Test manual PTX code generation passes
        kernel_file = Path(__file__).parent / "sample_kernels" / "vec_add.py"
        with open(kernel_file, "r") as f:
            kernel_code = f.read()
        kernel_ast = ast.parse(tw.dedent(kernel_code))
        return kernel_ast

        tree = passes.remove_appy(tree)
        tree = passes.block_loop(tree)
        tree = passes.to_unit_stmts_form(tree)
        tree, type_map = passes.attach_types(tree, self.val_map)
        
        tree = passes.to_pseudo_ptx(tree, self.val_map, type_map)
        tree = passes.add_builtin_imports(tree)
        return tree
        