import os
import ast
import ast_comments as astc
import textwrap as tw
from pathlib import Path
import ast_transforms as at
from ..base import Backend
from ...utils import load_module_from_str

class TritonBackend(Backend):
    def codegen(self, tree, metadata):
        '''
        Generate Triton GPU code for a single for loop.
        '''
        print("Input AST:\n", ast.dump(tree))
        from .passes import sanity_check, parse_pragma
        sanity_check.visit(tree)
        pragma = parse_pragma.visit(tree)
        print("Pragma:\n", pragma)
        
        val_map = metadata['val_map']
        from .passes import gen_imports
        from .passes import gen_data_movement
        
        tree = gen_data_movement.transform(tree, val_map)
        tree = gen_imports.transform(tree)
        ast.fix_missing_locations(tree)
        return tree
        
    def exec(self, tree, namespace=None):
        src = astc.unparse(tree)
        print("Generated:\n", src)
        m = load_module_from_str(src, namespace)