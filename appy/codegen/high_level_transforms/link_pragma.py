import ast_comments as ast
from ast import unparse
from appy.ast_utils import *
from .utils import parse_pragma

class PragmaLinker(ast.NodeTransformer):
    def __init__(self):
        self.cur_loop_pragma = None
        self.cur_top_pragma = None
        self.pragma_dict = None
        self.verbose = False

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
                self.cur_loop_pragma = node.value
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
