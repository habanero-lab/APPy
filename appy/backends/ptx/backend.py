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
        print(self.val_map)
        
    def codegen(self, tree, metadata):
        tree = passes.block_loop(tree)        
        tree = passes.attach_types(tree, self.val_map)
        return tree
        