import re
from collections import OrderedDict

def slice_to_tuple(s):
    low, up = s.replace(' ', '').split(':')
    if low == '':
        low = 0
    if up == '':
        assert False, 'upper bound of the slice must be specified: ' + s
    return low, up

def parse_pragma(pragma):
    d = OrderedDict()
    s = pragma.replace('#pragma', '')
    for item in s.split(' '):
        if item == '':
            continue
        
        key, value = item.split('=>')            
        props = {'parallel': False, 'block': 1, 'in_reg': False, 'reduce': None}
        for prop in value.split(','):
            
            match = re.search(r'\((.*?)\)', prop)
            if match:
                arg = match.groups()[0]
                prop_name = prop.split('(')[0]
                props[prop_name] = arg
            else:
                props[prop] = True
        
        d[slice_to_tuple(key)] = props
    
    return d