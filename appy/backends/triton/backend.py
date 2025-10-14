import ast
import ast_transforms as at
from ..backend import Backend

class TritonBackend(Backend):
    def codegen(self, loop_source, metadata):
        tree = ast.parse(loop_source).body[0]
        used_names = at.get_used_names(tree, no_funcname=True)
        # Remove loop target name from used_names since it should be a local var regardless
        used_names = [x for x in used_names if x != tree.target.id]
        print(used_names)
        return loop_source