import ast
from .utils import is_numpy_array, is_torch_tensor

class AttachShape(ast.NodeVisitor):
    def __init__(self, val_map):
        self.val_map = val_map
        self.verbose = True

    def visit_Call(self, node):
        self.generic_visit(node)
    
    def visit_Name(self, node):
        node.shape = []
        if node.id in self.val_map:
            val = self.val_map[node.id]
            if is_numpy_array(val) or is_torch_tensor(val):
                shape = val.shape
                # Using array expression of dimensionality > 1 is not supported
                if len(shape) > 1:
                    raise RuntimeError(f"Using array expression of dimensionality > 1 is not supported: {node.id}")
                node.shape = [("0", f"{s}") for s in shape]

        if self.verbose:
            print(f"[Attach Shape] attach shape {node.shape} to {ast.unparse(node)}")

    def visit_Constant(self, node):
        node.shape = []

    def get_sliced_shape(self, upper, slice: ast.Slice):
        lower = "0"
        upper = f"{upper}"
        if hasattr(slice, 'lower') and slice.lower is not None:
            lower = ast.unparse(slice.lower)
        if hasattr(slice, 'upper') and slice.upper is not None:
            upper = ast.unparse(slice.upper)
        if hasattr(slice, 'step') and slice.step is not None:
            raise RuntimeError(f"Step is not supported in slicing: {ast.unparse(slice)}")
        return (lower, upper)
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        shapes = [child.shape for child in [node.left, node.right] if child.shape]
        
        if shapes:
            # Make sure all shape in shapes are the same
            assert all([s == shapes[0] for s in shapes]), f"Shapes are not the same: {shapes}"
            node.shape = shapes[0]
        else:
            node.shape = []

        if self.verbose:
            print(f"[Attach Shape] attach shape {node.shape} to {ast.unparse(node)}")

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        shapes = [child.shape for child in [node.left] + node.comparators if child.shape]
        
        if shapes:
            # Make sure all shape in shapes are the same
            assert all([s == shapes[0] for s in shapes]), f"Shapes are not the same: {shapes}"
            node.shape = shapes[0]
        else:
            node.shape = []

        if self.verbose:
            print(f"[Attach Shape] attach shape {node.shape} to {ast.unparse(node)}")

    def visit_IfExp(self, node: ast.IfExp):
        self.generic_visit(node)
        shapes = [child.shape for child in [node.body, node.orelse] if child.shape]
        
        if shapes:
            # Make sure all shape in shapes are the same
            assert all([s == shapes[0] for s in shapes]), f"Shapes are not the same: {shapes}"
            node.shape = shapes[0]
        else:
            node.shape = []

        if self.verbose:
            print(f"[Attach Shape] attach shape {node.shape} to {ast.unparse(node)}")

    def visit_Call(self, node):
        self.generic_visit(node)
        shapes = [child.shape for child in node.args if child.shape]
        
        if shapes:
            # Make sure all shape in shapes are the same
            assert all([s == shapes[0] for s in shapes]), f"Shapes are not the same: {shapes}"
            node.shape = shapes[0]
        else:
            node.shape = []

        if self.verbose:
            print(f"[Attach Shape] attach shape {node.shape} to {ast.unparse(node)}")

    
    def visit_Subscript(self, node):        
        # Don't visit node.value
        assert isinstance(node.value, ast.Name), f"Unsupported type: {ast.unparse(node)}"
        self.visit(node.slice)
    
        if node.value.id in self.val_map:
            val = self.val_map[node.value.id]
            slice = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
            
            if is_numpy_array(val) or is_torch_tensor(val):
                shape = list(val.shape)
                
                sliced_shape = []
                for idx in slice:
                    # If idx is not a slice, pop one dimension from shape (scalar indexing)
                    # If idx is a slice, compute the sliced dim from shape
                    if isinstance(idx, ast.Slice):
                        dim = shape.pop(0)
                        sliced_shape.append(self.get_sliced_shape(dim, idx))
                        
                    else:
                        shape.pop(0)
              
            else:
                assert False, "Unsupported type: " + type(val).__name__
            
            node.shape = sliced_shape
            if self.verbose:
                print(f"[Attach Shape] attach shape {node.shape} to {ast.unparse(node)}")
        else:
            raise RuntimeError(f"Only arrays defined outside of parallel region can be sliced: {ast.unparse(node)}")

                
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
    
    def visit_Assign(self, node):
        # Clear all the in-flight loop info
        self.loop = None
        self.loop_bounds = None

        self.generic_visit(node)
        # Check if a loop has been generated
        if self.loop:
            print(f"loop bound: {self.loop_bounds}")
            print(f"target shape: {node.targets[0].shape}")
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

    def visit_BinOp(self, node):
        shapes = [child.shape for child in [node.left, node.right] if child.shape]
        if shapes:
            print("array expansion needed for", ast.unparse(node))
            shape = shapes[0]
            assert len(shape) == 1, "Only 1D array expansion is supported"
            print(shape)
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
            
            # self.loop.body.append(
            #     ast.Assign(
            #         targets=[ast.Name(id=self.get_new_temp_var(), ctx=ast.Store())],
            #         value=node
            #     )
            # )
            
        return node


def transform(tree, val_map):
    '''
    This pass detects and rewrites tensor expressions to explicit loops using a bottom-up traversal.
    '''
    AttachShape(val_map).visit(tree)
    tree = LowerToLoop().visit(tree)
 
    return tree