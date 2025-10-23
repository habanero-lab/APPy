import ast

class ToPseudoPTX(ast.NodeTransformer):
    def __init__(self, val_map, type_map):
        self.val_map = val_map
        self.type_map = type_map
        self.u32_reg_count = 0
        self.u64_reg_count = 0
        self.f32_reg_count = 0
        self.f64_reg_count = 0
        self.reg_map = {}  # Map from variable names to register names

    def insert_ld_param(self, node):
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
            target = ""
            func = ""
            if ty == "int32":
                target = f"__r{self.u32_reg_count}"
                self.u32_reg_count += 1
                func = "ptx_ld_param_u32"
            elif ty == "float32":
                target = f"__f{self.f32_reg_count}"
                self.f32_reg_count += 1
                func = "ptx_ld_param_f32"
            elif ty == "float64":
                target = f"__fd{self.f64_reg_count}"
                self.f64_reg_count += 1
                func = "ptx_ld_param_f64"
            elif ty.endswith('*'):
                target = f"__rd{self.u64_reg_count}"
                self.u64_reg_count += 1
                func = "ptx_ld_param_u64"
            else:
                raise NotImplementedError(f"LD param not implemented for type {ty}")
            
            self.reg_map[var] = target
            assign = ast.Assign(
                targets=[ast.Name(id=target, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id=func, ctx=ast.Load()),
                    args=[ast.Name(id=var, ctx=ast.Load())],
                    keywords=[]
                )
            )
            assigns.append(assign)
        node.body = assigns + node.body

    def visit_For(self, node):
        self.insert_ld_param(node)
        self.generic_visit(node)
        ast.fix_missing_locations(node)
        return node