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
    s = pragma.replace('#pragma', '').replace('simd', 'block')
    for item in s.split(' '):
        if item == '':
            continue
        
        key, value = item.split('=>')            
        props = {'parallel': False, 'block': 1, 'single_block': False, 'reduce': None}
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

def get_pragma_property(pragma, property_name):
    match = re.search(r' ' + property_name + r'\((.*?)\)', pragma)
    if match:
        p = match.groups()[0]
        return p
    else:
        return None