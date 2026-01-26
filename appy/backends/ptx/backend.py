import os
import ast
import tempfile
import textwrap as tw
from pathlib import Path
from ..base import Backend
from . import passes as passes

class PTXBackend(Backend):            
    def codegen(self, tree, metadata):
        val_map = metadata['val_map']
        tree = passes.remove_appy(tree)
        tree = passes.block_loop(tree)
        tree = passes.to_unit_stmts_form(tree)
        tree, type_map = passes.attach_types(tree, val_map)
        ptx_code = passes.codegen_ptx(tree, val_map, type_map)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as temp:
            temp.write(ptx_code)
            
            # You can get the file's name if you need to pass it
            # to another process or function.
            print(f"File created at: {temp.name}")
            
            # The file is guaranteed to exist inside this block.
            # You can .flush() if you need to ensure it's on disk *right now*.
            temp.flush()
       
        print("Generated PTX Code:\n", ptx_code)
                
        tree = passes.codegen_kernel_launch(tree)
        tree = passes.codegen_data_movement(tree, val_map)
        tree = passes.codegen_load_kernel(tree, temp.name)
        tree = passes.codegen_pycuda_imports(tree)
        
        ast.fix_missing_locations(tree)
        return tree
        
    def exec(self, tree, namespace=None):
        obj = compile(tree, filename=f"<ast>", mode="exec")
        exec(obj, namespace)