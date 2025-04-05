import ast_comments as ast
from ast import unparse
from appy.ast_utils import *
from .utils import *
import re
from collections import OrderedDict

# Code by ChatGPT :)
def parse_pragma(pragma_str):
    # Remove "#pragma" and any trailing colon
    pragma_str = pragma_str.strip().lstrip("#pragma").strip()
    
    tokens = []
    i = 0
    while i < len(pragma_str):
        if pragma_str[i].isspace():
            i += 1
            continue
        # Match a clause with parentheses like to(a,b)
        if match := re.match(r'(\w+)\s*\(([^)]*)\)', pragma_str[i:]):
            key, val = match.group(1), match.group(2)
            val_tuple = tuple(v.strip() for v in val.split(',') if v.strip())
            tokens.append((key, val_tuple if len(val_tuple) > 1 else val_tuple[0]))
            i += match.end()
        else:
            # Match single-word tokens like 'parallel' or 'simd'
            match = re.match(r'\w+', pragma_str[i:])
            if match:
                tokens.append((match.group(0), True))
                i += match.end()
            else:
                raise ValueError(f"Unexpected syntax at: {pragma_str[i:]}")
    
    # Postprocess to merge 'parallel for' into 'parallel_for'
    result = OrderedDict()
    skip_next = False
    for idx, (key, val) in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if key == 'parallel' and idx + 1 < len(tokens) and tokens[idx + 1][0] == 'for':
            result['parallel_for'] = True
            skip_next = True
        else:
            result[key] = val

    return result


class PragmaLinker(ast.NodeTransformer):
    def __init__(self):
        self.cur_loop_pragma = None
        self.cur_top_pragma = None
        self.verbose = False

    def convert_simd_directive(self, pragma):
        if ' simd' in pragma:
            arg = get_pragma_property(pragma, 'simd')
            if arg:
                pragma = pragma.replace(' simd', ' block')
            else:
                # Compiler determines a block size
                if 'reduction' in pragma:
                    pragma = pragma.replace(' simd', ' block(1024)')
                else:
                    pragma = pragma.replace(' simd', ' block(256)')
        return pragma

    def convert_le_prop(self, pragma):
        if 'le(' in pragma:
            #print(pragma)
            i = pragma.find('le(') + len('le(')
            size = ''
            while pragma[i] != ')':
                size += pragma[i]
                i += 1
            
            pragma = pragma.replace(f'le({size})', f'block({size}),single_block')
            #print(pragma)
            
        return pragma

    def visit_Comment(self, node):
        comment = node.value.strip()
        if comment.startswith('#pragma '):
            if comment == '#pragma parallel':
                comment += ' for'

            if comment.startswith('#pragma parallel for') or \
                comment.startswith('#pragma simd') or\
                comment.startswith('#pragma sequential for'):
                self.cur_loop_pragma = self.convert_simd_directive(node.value)
            else:
                self.cur_top_pragma = self.convert_le_prop(node.value)
            return None
        else:
            return node

    def visit_Assign(self, node): 
        #dump(node)       
        pragma = self.cur_top_pragma
        
        if pragma:
            node.pragma = pragma
            self.cur_top_pragma = None
            if self.verbose:
                print(f'associated `{unparse(node)}` with pragma `{node.pragma}`')
        return node

    def visit_For(self, node):
        pragma = self.cur_loop_pragma
        if pragma:
            node.pragma = pragma
            node.pragma_dict = parse_pragma(pragma)
            self.cur_loop_pragma = None
            if self.verbose:
                print(f'associated `{unparse(node)}` with pragma `{node.pragma}`')
        self.generic_visit(node)
        return node
