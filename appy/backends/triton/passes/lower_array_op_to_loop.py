import ast
from ....ast_utils import is_call

class RewriteReductionAssign(ast.NodeTransformer):
    def visit_Assign(self, node):
        self.generic_visit(node)

        if isinstance(node, ast.Call) and ast.unparse(node.func) in ['sum', 'min', 'max'] and node.args[0].shape:
            # This is reducing an array expression
            # If is 'sum', rewrite the assignment to be 3 assigns:
            #     __tmp_var = 0.0
            #     __tmp_var = __tmp_var + node.args[0]
            #     target = __tmp_var
            # Similarly, for min and max, rewrite to 3 assigns:
            #     __tmp_var = float('inf')
            #     __tmp_var = min(__tmp_var, node.args[0])
            #     target = __tmp_var
            # 

            # node.args[0] will be scalarized by the next pass
            pass
        return node

class ReplaceSliceWithVar(ast.NodeTransformer):
    '''
    A simple replacement for every single slice - don't even need to check the slice bounds.
    '''
    def __init__(self, var):
        self.var = var

    def visit_Slice(self, node):
        return ast.Name(id=self.var, ctx=ast.Load())

class LowerToLoop(ast.NodeTransformer):
    def __init__(self):
        self.loop = None # loop in-flight
        self.loop_bounds = None
        self.loop_index = None
        self.loop_index_count = 0
        self.temp_var_count = 0

    def get_new_loop_index(self):
        name = f"__a2l_i{self.loop_index_count}"
        self.loop_index_count += 1
        return name
    
    def get_new_temp_var(self):
        name = f"__a2l_t{self.temp_var_count}"
        self.temp_var_count += 1
        return name
    
    def gen_reduction(self, node):
        raise NotImplementedError("Lowering reduction array ops to loops is to be implemented:\n" + ast.unparse(node))
    
    def visit_Assign(self, node):
        # Clear all the in-flight loop info
        self.loop = None
        self.loop_bounds = None

        self.generic_visit(node)
        # Check if a loop has been generated
        if self.loop:
            print(f"loop bound: {self.loop_bounds}")
            print(f"target shape: {node.targets[0].shape}")
            if is_call(node.value, ["sum", "min", "max"]):
                return self.gen_reduction(node)
            
            # If so, check if the target also has matching shape
            assert len(node.targets[0].shape) == 1, "Only 1D array expansion is supported, but got shape: " + str(node.targets[0].shape)
            if node.targets[0].shape[0] != self.loop_bounds:
                raise RuntimeError(f"Target shape and loop bound are not the same: {node.targets[0].shape} vs {self.loop_bounds}")
            
            node.targets[0] = ReplaceSliceWithVar(self.loop.target.id).visit(node.targets[0])

            # Add to loop body
            self.loop.body.append(node)
            self.loop.pragma = {"simd": True}
            
            ast.fix_missing_locations(self.loop)
            print(ast.unparse(self.loop))
            return self.loop
            
        return node
    
    def visit_Subscript(self, node):
        shape = node.shape
        if shape:
            print("array expansion needed for", ast.unparse(node))
            assert len(shape) == 1, "Only 1D array expansion is supported"
            #print(shape)
            low, up = shape[0]
            # Generate a for loop with range(low, up)
            if not self.loop:
                self.loop = ast.For(
                    target=ast.Name(id=self.get_new_loop_index(), ctx=ast.Store()),
                    iter=ast.Call(
                        func=ast.Name(id='range', ctx=ast.Load()),
                        args=[ast.parse(low).body[0].value, ast.parse(up).body[0].value, ast.Constant(1)],
                        keywords=[]
                    ),
                    body=[],
                    orelse=[],
                )
                self.loop_bounds = (low, up)
                
            node = ReplaceSliceWithVar(self.loop.target.id).visit(node)
            
        return node


def transform(tree):
    '''
    This pass detects and rewrites tensor expressions to explicit loops using a bottom-up traversal.
    '''
    tree = LowerToLoop().visit(tree)
    return tree