import ast
from .utils import is_numpy_array, is_torch_tensor

class AttachShape(ast.NodeVisitor):
    def __init__(self, val_map):
        self.val_map = val_map
        self.verbose = False

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

                


def visit(tree, val_map):
    AttachShape(val_map).visit(tree)