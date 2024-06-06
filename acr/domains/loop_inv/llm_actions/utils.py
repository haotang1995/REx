#!/usr/bin/env python
# coding=utf-8

import re
import z3
import copy
from openai import ChatCompletion
from ....utils.llm.utils import extract_code_blocks, eval_code
from ..data.z3_checks import init_exec_globals, find_vars, init_vars, eval_expr, invtoinv2

def is_valid_z3(code):
    if 'Lambda' in code:
        return False
    if 'z3 ' in code:
        return False
    try:
        exec_globals = init_exec_globals()
        all_vars = find_vars(code)
        all_vars.update(set([v+'2' for v in all_vars]))
        init_vars(all_vars, exec_globals)
        eval_expr('hha', code, exec_globals)
        inv2 = invtoinv2(z3.simplify(exec_globals['hha']), exec_globals, all_vars)
        return isinstance(exec_globals['hha'], z3.z3.BoolRef)
    except Exception as e:
        return False

def preprocess_code_block(code_block):
    return '\n'.join([preprocess_line(line) for line in code_block.split('\n')])
def preprocess_line(line):
    line = line.strip()
    if ' = ' in line:
        line = line[line.index(' = ') + 3:]
    if line.startswith('#') or line.startswith('//'):
        words = line.split()
        if len(words) > 1:
            line = line[line.index(words[1]):]
    if '&&' in line:
        return 'And(%s)' % line.replace('&&', ',')
    elif '||' in line:
        return 'Or(%s)' % line.replace('||', ',')
    return line

def remove_comment(block):
    return '\n'.join([
        line for line in block.split('\n')
        if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('//')
    ])
def split_code_blocks(code):
    code = remove_comment(code)
    blocks = []
    prv_index = 0
    lines = code.split('\n')
    for index in range(len(lines)):
        if index < len(lines)-1 and len(lines[index+1].lstrip()) != len(lines[index+1]):
            continue
        if index < len(lines)-1 and (lines[index].startswith('#') or lines[index].startswith('//')):
            continue
        if index < len(lines)-1 and (lines[index+1].startswith('else:') or lines[index+1].startswith('elif ')):
            continue
        _local_exec_globals = eval_code('\n'.join(lines[:index+1]), exec_globals={}, return_exec_globals=True)
        if not isinstance(_local_exec_globals, dict):
            continue
        blocks.append('\n'.join(lines[prv_index:index+1]))
        prv_index = index + 1
    if prv_index < len(lines):
        blocks.append('\n'.join(lines[prv_index:]))
    return blocks

def copyable(v):
    try:
        copy.deepcopy(v)
        return True
    except:
        return False

def _replace(s, start, end):
    if start not in s:
        return s
    sindex = s.index(start)
    if end not in s[sindex + len(start):]:
        return s
    eindex = sindex + len(start) + s[sindex + len(start):].index(end)
    return _replace(s[:sindex] + s[sindex + len(start):eindex] + s[eindex + len(end):], start, end)
def extract_loop_invariants(completion):
    assert len(completion.choices) == 1

    output = completion.choices[0].message.content
    code_blocks = extract_code_blocks(output)
    code_blocks = [preprocess_code_block(block) for block in code_blocks] + [block for code in code_blocks for block in split_code_blocks(code)]

    # Replace Int('x') with x
    code_blocks = [_replace(block, 'Int("', '")') for block in code_blocks]
    code_blocks = [_replace(block, "Int('", "')") for block in code_blocks]
    code_blocks = [_replace(block, 'z3.Int("', '")') for block in code_blocks]
    code_blocks = [_replace(block, "z3.Int('", "')") for block in code_blocks]
    code_blocks = list(set(code_blocks))

    valids = [is_valid_z3(block) for block in code_blocks]
    valid_code_blocks = [block for block, valid in zip(code_blocks, valids) if valid]
    for block, valid in zip(code_blocks, valids):
        if not valid:
            lines = block.split('\n')
            for line in lines:
                if is_valid_z3(line):
                    valid_code_blocks.append(line)
                if is_valid_z3(line.strip().strip(',').strip()):
                    valid_code_blocks.append(line.strip().strip(',').strip())

    return valid_code_blocks

if __name__ == '__main__':
    class DotObject:
        def __init__(self, name, value):
            self.name = name
            self.value = value
            self.__dict__[name] = value
    content = '''
```python
And(
    Not(x == 0),
    Not(y == 0),
    u == b * Int('z1'),
    v == a * Int('z2'),
    x * Int('z3') == a * Int('z1'),
    y * Int('z4') == b * Int('z2'),
    Int('z1') >= 0,
    Int('z2') >= 0,
    Int('z3') >= 0,
    Int('z4') >= 0
)
```
    '''
    message = DotObject('content', content)
    choice = DotObject('message', message)
    completion = DotObject('choices', [choice])

    loops = extract_loop_invariants(completion)
    print(loops)
    for li, loop in enumerate(loops):
        assert is_valid_z3(loop), (f'Loop invariant {li} is not valid Z3 code', loop, is_valid_z3(loop))
