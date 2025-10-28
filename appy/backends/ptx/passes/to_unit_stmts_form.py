import ast

class ExpandAssign(ast.NodeTransformer):
    '''
    This pass takes one assignment as input and returns a list of unit statements
    after calling its visit() method.
    '''
    def __init__(self, counter):
        self.new_stmts = []
        self.counter = counter

    def get_temp_var(self):
        temp_var = f"__tmp_{self.counter}"
        self.counter += 1
        return temp_var

    def visit_Assign(self, node):
        self.generic_visit(node)
        if self.new_stmts:
            last_stmt = self.new_stmts[-1]
            last_target = last_stmt.targets[0]
            # Combine the last statement with the assignment if possible
            if isinstance(last_target, ast.Name) and isinstance(node.targets[0], ast.Name) and last_target.id == node.value.id:                
                node.value = last_stmt.value
                self.new_stmts.pop()

            for stmt in self.new_stmts:
                ast.fix_missing_locations(stmt)
        return self.new_stmts + [node]
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        temp_var = self.get_temp_var()
        assign = ast.Assign(
            targets=[ast.Name(id=temp_var, ctx=ast.Store())],
            value=node
        )
        self.new_stmts.append(assign)
        return ast.Name(id=temp_var, ctx=ast.Load())
    
    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Store):
            # For store operations, we don't need to create a temp variable
            return node
        
        temp_var = self.get_temp_var()
        assign = ast.Assign(
            targets=[ast.Name(id=temp_var, ctx=ast.Store())],
            value=node
        )
        self.new_stmts.append(assign)
        return ast.Name(id=temp_var, ctx=ast.Load())
    
    def visit_Compare(self, node):
        self.generic_visit(node)
        temp_var = self.get_temp_var()
        assign = ast.Assign(
            targets=[ast.Name(id=temp_var, ctx=ast.Store())],
            value=node
        )
        self.new_stmts.append(assign)
        return ast.Name(id=temp_var, ctx=ast.Load())
    
    def visit_Call(self, node):
        self.generic_visit(node)
        temp_var = self.get_temp_var()
        assign = ast.Assign(
            targets=[ast.Name(id=temp_var, ctx=ast.Store())],
            value=node
        )
        self.new_stmts.append(assign)
        return ast.Name(id=temp_var, ctx=ast.Load())
    

class ToUnitStmtsForm(ast.NodeTransformer):
    '''
    This pass expands compound expressions such as A[i] + B[i] into unit statements 
    where a statement contains at most one of the following:
        * An arithmetic operation
        * A math function call
        * A logical operation
        * A comparison
        * A bitwise operation
        * An array load or store 
    '''
    def __init__(self):
        self.var_counter = 0

    def visit_Assign(self, node):
        expander = ExpandAssign(self.var_counter)
        new_nodes = expander.visit_Assign(node)
        self.var_counter = expander.counter
        return new_nodes
    
    def visit_If(self, node):
        self.generic_visit(node)
        return node
    