import ast

class ReplaceWithRegisters(ast.NodeTransformer):
    def __init__(self, val_map, type_map):
        self.val_map = val_map
        self.type_map = type_map
        self.u32_reg_count = 0
        self.u64_reg_count = 0
        self.f32_reg_count = 0
        self.f64_reg_count = 0
        self.reg_map = {}  # Map from variable names to register names

    def get_new_register(self, ty):
        if ty == "int32":
            reg = f"__r{self.u32_reg_count}"
            self.u32_reg_count += 1
            return reg
        elif ty == "float32":
            reg = f"__f{self.f32_reg_count}"
            self.f32_reg_count += 1
            return reg
        elif ty == "float64":
            reg = f"__fd{self.f64_reg_count}"
            self.f64_reg_count += 1
            return reg
        elif ty.endswith('*'):
            reg = f"__rd{self.u64_reg_count}"
            self.u64_reg_count += 1
            return reg
        else:
            raise NotImplementedError(f"Register allocation not implemented for type {ty}")
        
    def get_ld_func(self, ty):
        if ty == "int32":
            return "ptx_ld_param_u32"
        elif ty == "float32":
            return "ptx_ld_param_f32"
        elif ty == "float64":
            return "ptx_ld_param_f64"
        elif ty.endswith('*'):
            return "ptx_ld_param_u64"
        else:
            raise NotImplementedError(f"LD param not implemented for type {ty}")
        
    def get_reg_for_var(self, var):
        if var in self.reg_map:
            return self.reg_map[var]
        else:
            raise ValueError(f"Register for variable {var} not found.")
        
    def set_reg_for_var(self, var, reg):
        self.reg_map[var] = reg

    def create_ld_param(self, node):
        '''
        Create a new assignment for each var based on its type. 
        for type int32, do
            __r0 = ptx_ld_param_u32(var)
        for pointer types, do
            __r1 = ptx_ld_param_u64(var)
        for type float32, do
            __r2 = ptx_ld_param_f32(var)
        for type float64, do
            __r3 = ptx_ld_param_f64(var)
        The naming convention for registers is:
            __r for u32,
            __rd for u64,
            __f for f32,
            __fd for f64.
        '''
        assigns = []
        for var in self.val_map.keys():
            assert var in self.type_map, f"Type information missing for variable {var}"
            ty = self.type_map[var]
            func = self.get_ld_func(ty)
            register = self.get_new_register(ty)            
            self.set_reg_for_var(var, register)
            assign = ast.Assign(
                targets=[ast.Name(id=register, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id=func, ctx=ast.Load()),
                    args=[ast.Name(id=var, ctx=ast.Load())],
                    keywords=[]
                )
            )
            assigns.append(assign)
        return assigns

    def create_assigns_for_special_registers(self, node):
        register = self.get_new_register("int32")  # for __ctaid_x
        self.set_reg_for_var("__ctaid_x", register)
        assign = ast.Assign(
            targets=[ast.Name(id=register, ctx=ast.Store())],
            value=ast.Name(id="__ctaid_x", ctx=ast.Load())
        )
        return [assign] 

    def visit_For(self, node):
        assigns = []
        assigns.extend(self.create_ld_param(node))
        assigns.extend(self.create_assigns_for_special_registers(node))
        for child in node.body:
            self.visit(child)
        node.body = assigns + node.body
        ast.fix_missing_locations(node)
        return node
    
    def visit_Call(self, node):
        for i, arg in enumerate(node.args):
            self.visit(arg)
        return node
    
    def visit_Name(self, node):
        if self.is_register(node.id):
            return node  # already a register, do nothing
            
        assert node.id in self.reg_map, f"Variable {node.id} not found in register map."
        reg = self.get_reg_for_var(node.id)
        new_node = ast.Name(id=reg, ctx=node.ctx)
        ast.copy_location(new_node, node)
        return new_node
    
    def is_not_register(self, name):
        return not name.startswith("__r") and not name.startswith("__f") and not name.startswith("__rd") and not name.startswith("__fd")
    
    def is_register(self, name):
        return not self.is_not_register(name)

    def visit_Assign(self, node):
        target = node.targets[0]
        # Assign a new register if needed
        if isinstance(target, ast.Name) and self.is_not_register(target.id) and target.id not in self.reg_map:
            ty = self.type_map[target.id]
            reg = self.get_new_register(ty)
            self.set_reg_for_var(target.id, reg)
        self.generic_visit(node)
        return node

    
def transform(tree, val_map, type_map):
    return ReplaceWithRegisters(val_map, type_map).visit(tree)