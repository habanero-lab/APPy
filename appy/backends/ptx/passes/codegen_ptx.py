import ast
from .ptx_types import PTXType

class CodegenPTX(ast.NodeTransformer):
    def __init__(self, val_map, type_map):
        self.val_map = val_map
        self.type_map = type_map
        self.ptx_code = ""
        self.b32_reg_count = 0
        self.b64_reg_count = 0
        self.f32_reg_count = 0
        self.f64_reg_count = 0
        self.var_to_reg_map = {
            '__ctaid_x': r'%ctaid.x'
        }

    def get_new_b32_reg(self):
        reg_name = f"%r{self.b32_reg_count}"
        self.b32_reg_count += 1
        return reg_name
    
    def get_new_b64_reg(self):
        reg_name = f"%rd{self.b64_reg_count}"
        self.b64_reg_count += 1
        return reg_name
    
    def get_new_f32_reg(self):
        reg_name = f"%f{self.f32_reg_count}"
        self.f32_reg_count += 1
        return reg_name 
    
    def get_new_f64_reg(self):
        reg_name = f"%fd{self.f64_reg_count}"
        self.f64_reg_count += 1
        return reg_name

    def gen_header(self):
        self.ptx_code += ".version 5.0\n"
        self.ptx_code += ".target sm_50\n"
        self.ptx_code += ".address_size 64\n\n"

    def gen_func_decl(self):
        self.ptx_code += ".visible .entry kernel(\n"
        param_lines = []
        for var, ptx_type in self.type_map.items():
            if var in self.val_map:
                param_lines.append(f"    .param .{ptx_type.value} param_{var}")
        self.ptx_code += ",\n".join(param_lines)
        self.ptx_code += "\n)\n{\n"

    def gen_func_closure(self):
        self.ptx_code += "}\n"

    def gen_ld_params(self):
        for var, ptx_type in self.type_map.items():
            # These are parameters to be loaded
            if var in self.val_map:
                reg = self.get_reg_for_var(var)
                self.ptx_code += f"    ld.param.{ptx_type.value} {reg}, [param_{var}];\n"

    def record_var_to_reg_map(self, var, reg):
        if var not in self.var_to_reg_map:
            self.var_to_reg_map[var] = reg
        else:
            assert self.var_to_reg_map[var] == reg, f"Variable {var} mapped to different registers {self.var_to_reg_map[var]} and {reg}"

    def get_reg_for_var(self, var):
        if var in self.var_to_reg_map:
            return self.var_to_reg_map[var]
        else:
            # Create a new register for this variable
            ptx_type = self.type_map[var]
            reg = ""
            if ptx_type == PTXType.U32 or ptx_type == PTXType.S32:
                reg = self.get_new_b32_reg()
            elif ptx_type == PTXType.U64 or ptx_type == PTXType.S64:
                reg = self.get_new_b64_reg()
            elif ptx_type == PTXType.F32:
                reg = self.get_new_f32_reg()
            elif ptx_type == PTXType.F64:
                reg = self.get_new_f64_reg()
            self.record_var_to_reg_map(var, reg)
            return reg
        
    def var_has_reg(self, var):
        return var in self.var_to_reg_map

    def get_ptx_code(self):
        return self.ptx_code.strip()

    def visit_For(self, node):
        self.gen_header()
        self.gen_func_decl()
        self.gen_ld_params()
        self.generic_visit(node)
        self.gen_func_closure()
        return []
    
    def gen_assign_const_to_var(self, target, value):
        var_name = target.id
        dest_reg = self.get_reg_for_var(var_name)
        op_type = self.type_map[var_name].value
        if op_type in ['s32', 'u32']:
            self.ptx_code += f"    mov.{op_type} {dest_reg}, {value.value};\n"
        elif op_type in ['s64', 'u64']:
            self.ptx_code += f"    mov.{op_type} {dest_reg}, {value.value}l;\n"
        elif op_type in ['f32']:
            self.ptx_code += f"    mov.{op_type} {dest_reg}, {value.value}f;\n"
        elif op_type in ['f64']:
            self.ptx_code += f"    mov.{op_type} {dest_reg}, {value.value}d;\n"
        else:
            raise NotImplementedError(f"Constant assignment not implemented for type {op_type}")
        
    def gen_assign_var_to_var(self, target, value):
        src_var = value.id
        dest_var = target.id
        src_reg = self.get_reg_for_var(src_var)
        dest_reg = self.get_reg_for_var(dest_var)
        op_type = self.type_map[dest_var].value
        self.ptx_code += f"    mov.{op_type} {dest_reg}, {src_reg};\n"

    def gen_assign_binop_to_var(self, target, left, op, right):
        dest_var = target.id
        dest_reg = self.get_reg_for_var(dest_var)
        left_var = left.id
        right_var = right.id
        left_reg = self.get_reg_for_var(left_var)
        right_reg = self.get_reg_for_var(right_var)
        op_type = self.type_map[dest_var].value
        if isinstance(op, ast.Add):
            self.ptx_code += f"    add.{op_type} {dest_reg}, {left_reg}, {right_reg};\n"
        elif isinstance(op, ast.Sub):
            self.ptx_code += f"    sub.{op_type} {dest_reg}, {left_reg}, {right_reg};\n"
        elif isinstance(op, ast.Mult):
            self.ptx_code += f"    mul.{op_type} {dest_reg}, {left_reg}, {right_reg};\n"
        elif isinstance(op, ast.Div):
            self.ptx_code += f"    div.{op_type} {dest_reg}, {left_reg}, {right_reg};\n"
        else:
            raise NotImplementedError(f"Binary operation {type(op)} not implemented")
        
    def gen_offset_compute(self, subscript, op_type):
        base_reg = self.get_reg_for_var(subscript.value.id)
        slice_reg = self.get_reg_for_var(subscript.slice.id)
        addr_reg = self.get_new_b64_reg()
        # Generate a mul.wide.s32 instruction where the stride is 4 or 8 bytes depending on the type
        if op_type in ['s32', 'u32', 'f32']:
            self.ptx_code += f"    mad.wide.u32 {addr_reg}, {slice_reg}, 4, {base_reg};\n"
        elif op_type in ['s64', 'u64', 'f64']:
            self.ptx_code += f"    mad.wide.u32 {addr_reg}, {slice_reg}, 8, {base_reg};\n"
        return addr_reg

    def gen_assign_subscript_to_var(self, target, subscript): 
        # Generate ld.global instructions where offset needs to be computed
        addr_reg = self.gen_offset_compute(subscript, self.type_map[target.id].value)
        dest_reg = self.get_reg_for_var(target.id)
        op_type = self.type_map[target.id].value
        self.ptx_code += f"    ld.global.{op_type} {dest_reg} [{addr_reg}];\n"

    def gen_assign_var_to_subscript(self, subscript, name):
        val = self.get_reg_for_var(name.id)
        addr_reg = self.gen_offset_compute(subscript, self.type_map[name.id].value)
        op_type = self.type_map[name.id].value
        self.ptx_code += f"    st.global.{op_type} [{addr_reg}] {val};\n"
    
    def visit_Assign(self, node):
        assert len(node.targets) == 1, "Only single target assignments are supported"
        target = node.targets[0]
        if isinstance(target, ast.Name):
            if isinstance(node.value, ast.Constant):
                self.gen_assign_const_to_var(target, node.value)
                return node
            elif isinstance(node.value, ast.Name):
                self.gen_assign_var_to_var(target, node.value)
                return node
            elif isinstance(node.value, ast.BinOp):
                self.gen_assign_binop_to_var(target, node.value.left, node.value.op, node.value.right)
                return node
            elif isinstance(node.value, ast.Subscript):
                self.gen_assign_subscript_to_var(target, node.value)
                return node
            else:
                raise NotImplementedError(f"Only Constant, Name, and Subscript assignments are supported as values: {type(node.value)}")
            
        elif isinstance(target, ast.Subscript):
            if isinstance(node.value, ast.Name):
                self.gen_assign_var_to_subscript(target, node.value)
            else:
                assert False
        else:
            raise NotImplementedError("Only Name and Subscript targets are supported in assignments")
        