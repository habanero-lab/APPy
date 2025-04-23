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
        if '=>' not in item:
            continue
        
        key, value = item.split('=>')            
        props = {'parallel': False, 'block': 1, 'single_block': False, 'reduce': None}
        for prop in value.split(','):
            # Check if the property has an optional value, if so map it to the value, 
            # otherwise map it to True
            match = re.search(r'\((.*?)\)', prop)
            if match:
                arg = match.groups()[0]
                prop_name = prop.split('(')[0]
                props[prop_name] = arg
                if arg.isdigit():
                    props[prop_name] = int(arg)
            else:
                props[prop] = True
        
        d[slice_to_tuple(key)] = props
    
    return d

def get_default_block_size():
    return 256

def get_pragma_property(pragma, property_name):
    match = re.search(r' ' + property_name + r'\((.*?)\)', pragma)
    if match:
        p = match.groups()[0]
        return p
    else:
        return None

def has_tensor_pragma(node):
    if not hasattr(node, 'pragma'):
        return False
    else:
        pragma = node.pragma
        return '=>' in pragma

def has_atomic_pragma(node):
    if not hasattr(node, 'pragma'):
        return False
    else:
        pragma = node.pragma
        return '#pragma atomic' in pragma
    
    
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