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
        tree, type_map = passes.attach_types(tree, self.val_map)
        tree = passes.to_pseudo_ptx(tree, self.val_map, type_map)
        return tree
        