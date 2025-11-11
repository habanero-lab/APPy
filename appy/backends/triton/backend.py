import os
import ast
import ast_comments as astc
import textwrap as tw
import ast_transforms as at
from ..base import Backend
from ...utils import load_module_from_str

class TritonBackend(Backend):
    def codegen(self, tree, metadata):
        '''
        Generate Triton GPU code for a single for loop.
        '''
        #print("Input AST:\n", ast.dump(tree))
        from .passes import sanity_check, parse_pragma, rewrite_range, block_loop
        sanity_check.visit(tree)
        pragma = parse_pragma.visit(tree)
        tree = rewrite_range.transform(tree)
        tree = block_loop.transform(tree, pragma)
        
        val_map = metadata['val_map']
        from .passes import gen_imports
        from .passes import gen_data_movement
        from .passes import gen_kernel_launch
        from .passes import gen_kernel
        
        tree, h2d_map = gen_data_movement.transform(tree, val_map)
        tree = gen_kernel.transform(tree, val_map, metadata)
        tree = gen_kernel_launch.transform(tree, val_map, h2d_map, metadata)        

        # Add imports at last!
        tree = gen_imports.transform(tree)
        ast.fix_missing_locations(tree)
        return tree
        
    def exec(self, tree, namespace=None):
        src = astc.unparse(tree)        
        m = load_module_from_str(src, namespace)