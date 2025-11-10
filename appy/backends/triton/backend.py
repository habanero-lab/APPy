import os
import ast
import textwrap as tw
from pathlib import Path
import ast_transforms as at
from ..base import Backend
from ...utils import load_module_from_str

class TritonBackend(Backend):
    def codegen(self, tree, metadata):
        from .passes import gen_imports
        tree = gen_imports.transform(tree)
        return tree
        
    def exec(self, tree, namespace=None):
        src = ast.unparse(tree)
        print("Generated:\n", src)
        m = load_module_from_str(src, namespace)