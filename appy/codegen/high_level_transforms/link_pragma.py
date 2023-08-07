import ast_comments as ast
from ast import unparse
from appy.ast_utils import *


class PragmaLinker(ast.NodeVisitor):
    def __init__(self):
        self.cur_loop_pragma = None
        self.cur_top_pragma = None
        self.verbose = True

    def visit_Comment(self, node):
        
        if node.value.startswith('#pragma parallel'):
            self.cur_loop_pragma = node.value
        elif node.value.startswith('#pragma '):
            assert '=>' in node.value
            self.cur_top_pragma = node.value

    def visit_Assign(self, node):        
        pragma = self.cur_top_pragma
        if pragma:
            node.pragma = pragma
            self.cur_top_pragma = None
            if self.verbose:
                print(f'associated `{unparse(node)}` with pragma `{node.pragma}`')

    def visit_For(self, node):
        pragma = self.cur_loop_pragma
        if pragma:
            node.pragma = pragma
            self.cur_loop_pragma = None
            if self.verbose:
                print(f'associated `{unparse(node)}` with pragma `{node.pragma}`')