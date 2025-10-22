import os
import ast
import textwrap as tw
from pathlib import Path
import ast_transforms as at
from ..base import Backend

class PTXBackend(Backend):
    def codegen(self, tree, metadata):
        return tree
        