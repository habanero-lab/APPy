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
            
            if key in ['shared', 'to', 'from']:
                tokens.append((key, list(val_tuple)))
            else:
                tokens.append((key, val_tuple if len(val_tuple) > 1 else val_tuple[0]))
                
            # if len(val_tuple) == 0:
            #     tokens.append((key, val_tuple))
            # else:
            #     tokens.append((key, val_tuple if len(val_tuple) > 1 else val_tuple[0]))
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

    # Check for unrecognized clauses
    recognized_clauses = {'parallel_for', 'simd', 'block', 'reduction', 'shared', 'to', 'from', 'atomic'}
    for key in result:
        if key not in recognized_clauses:
            raise ValueError(f"Unrecognized pragma clause: `{key}` in `{pragma_str}`")

    return result

# Code by ChatGPT :)
def dict_to_pragma(pragma_dict):
    clauses = []

    # Handle 'parallel_for' specially, converting to 'parallel for'
    if pragma_dict.get('parallel_for'):
        clauses.extend(['parallel', 'for'])

    for key, val in pragma_dict.items():
        if key == 'parallel_for':
            continue  # already handled

        if val is True:
            clauses.append(key)
        elif isinstance(val, tuple):
            val_str = ', '.join(val)
            clauses.append(f'{key}({val_str})')
        else:
            clauses.append(f'{key}({val})')

    return '#pragma ' + ' '.join(clauses)

class PragmaLinker(ast.NodeTransformer):
    def __init__(self):
        self.cur_loop_pragma = None
        self.cur_top_pragma = None
        self.pragma_dict = None
        self.verbose = 1
    
    def process_simd_clause(self, pragma_dict):
        # Replace key `simd` with key `block`
        if 'simd' in pragma_dict:
            pragma_dict['block'] = pragma_dict['simd']
            del pragma_dict['simd']

            # If `block` has no value associated (by default it's True), set it to 256 or 1024
            if pragma_dict['block'] is True:
                if 'reduction' in pragma_dict:
                    pragma_dict['block'] = 1024
                else:
                    pragma_dict['block'] = 256
        return pragma_dict

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
            if '=>' in comment:
                self.cur_top_pragma = self.convert_le_prop(node.value)
            else:
                pragma_dict = parse_pragma(node.value)
                pragma_dict = self.process_simd_clause(pragma_dict)
                self.cur_loop_pragma = dict_to_pragma(pragma_dict)
                self.pragma_dict = pragma_dict
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
        if self.cur_loop_pragma:
            node.pragma = self.cur_loop_pragma
            node.pragma_dict = self.pragma_dict
            self.cur_loop_pragma = None
            if self.verbose:
                print(f'associated `{unparse(node)}` with pragma `{node.pragma}`')
        self.generic_visit(node)
        return node
