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

def visit(m):
    # Assume the first node is a pragma comment
    pragma = parse_pragma(m.body[0].value)
    return pragma