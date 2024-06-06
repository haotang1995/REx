#!/usr/bin/env python
# coding=utf-8

from ....utils.llm import LLM
from .utils import extract_loop_invariants

def _flat_str(s):
    lines = [l.strip() for l in s.strip().split('\n')]
    return ' '.join(lines)
def get_loop_invs_with_error_message(check_result, with_post_condition=False):
    simplified_loop_invs = check_result['intermediate']['simplified_loop_invariants']
    assert len(simplified_loop_invs) > 0
    assert set(simplified_loop_invs) != {'True'}
    loop_invs = 'And(\n' + ',\n'.join([f'    {inv}' for inv in simplified_loop_invs]) + '\n)'

    error_messages = 'The previous loop invariant is wrong because:\n\n'
    if not check_result['overall_metrics']['inductive']['success']:
        assert len(simplified_loop_invs) == len(check_result['per_inv_metrics'])
        for inv, mt in zip(simplified_loop_invs, check_result['per_inv_metrics']):
            inv = _flat_str(inv)
            if not mt['establish']['success'] and not mt['preserve']['success']:
                error_messages += f'  - `{inv}` is neither established nor preserved, meaning it is not even true in the beginning of the loop and is neither true after each iteration of the loop.'
            elif not mt['establish']['success']:
                error_messages += f'  - `{inv}` is not established, meaning it is not true in the beginning of the loop.'
            elif not mt['preserve']['success']:
                error_messages += f'  - `{inv}` is not preserved, meaning it is not true after each iteration of the loop.'
            if not mt['establish']['success']:
                if str(mt['establish']['_checker_result']) == 'unknown':
                    error_messages += f' The checker cannot determine whether `{inv}` is established due to timeout when checking the assertion `{mt["establish"]["assertion"]}`.'
                else:
                    assert mt['establish']['_checker_model'] is not None, (inv, mt)
                    model = _flat_str(mt['establish']['_checker_model'])
                    assertion = _flat_str(mt['establish']['assertion'])
                    error_messages += f' For example, we can set {model} to find a counterexample for establishing `{inv}`, since it conflicts with the assertion `{assertion}`.'
            elif not mt['preserve']['success']:
                if str(mt['preserve']['_checker_result']) == 'unknown':
                    error_messages += f' The checker cannot determine whether `{inv}` is preserved due to timeout when checking the assertion `{mt["preserve"]["assertion"]}`.'
                else:
                    assert mt['preserve']['_checker_model'] is not None, (inv, mt)
                    model = _flat_str(mt['preserve']['_checker_model'])
                    assertion = _flat_str(mt['preserve']['assertion'])
                    error_messages += f' For example, we can set {model} to find a counterexample for preserving `{inv}`, since it conflicts with the assertion `{assertion}`.'
            error_messages += '\n'
    if not check_result['overall_metrics']['post']['success']:
        error_messages += '  - The metrics are neither enough to imply the postconditions'
        if with_post_condition:
            error_messages += f' as {check_result["inputs"]["post"]}'
        error_messages += '.\n'

    return f'{loop_invs}\n\n{error_messages}'

def refine_loop_inv(data, check_result, llm, verbose_level=2):
    assert isinstance(llm, LLM)
    assert not check_result['success']

    code = data.code_for_llm()
    loop_invs_with_error_message = get_loop_invs_with_error_message(check_result)
    prompt = [
        {'role': 'system', 'content': '''
You are an expert software engineer and you are working on a project that requires you to write loop invariants for a loop in the code. The loop invariant is a condition that must be true before and after each iteration of the loop. The loop invariant helps you understand the behavior of the loop and ensures that the loop is working correctly. The loop invariant should be strong enough to prove the correctness of the loop as well as the functionality of the whole code snippet.
         '''.strip()},
        {'role': 'user', 'content': f'''
You are given the following code snippet in C:


```

{code}

```


Do you understand the code snippet? What function does the code snippet perform? Can you explain the code snippet part by part? What is the functionality of each part of the code snippet? What is the function of the loop in the code snippet?


The loop that needs a loop invariant has been identified in the code snippet with a line starting with `// loop invariant in Z3: `. Do you understand the function of the loop in the code snippet? Can you explain the function of the loop in the code snippet? What is the purpose of the loop in the code snippet?


What variables are there in the loop in the code snippet? How are the variables used and updated in the loop in the code snippet? What is the relationship among the variables in the loop in the code snippet?


According to those analysis, could you refine the following loop invariants that you generated before?


{loop_invs_with_error_message}


Please correct the previous loop invariants and provide the correct loop invariants for the loop in the code snippet. The loop invariant should be written in the format as Python-Z3 as before. Please surround the loop invariant with triple backticks and write it in the format as Python-Z3.
         '''.strip()},
    ]

    if verbose_level >= 2:
        print('Prompt:')
        for p in prompt:
            print('-'*5, f' Role: {p["role"]} ', '-'*20)
            print(p['content'])
        print()

    with llm.track() as costs:
        response = llm(prompt,)
    if verbose_level >= 2:
        print('-'*5, ' Response: ', '-'*20)
        print(response.choices[0].message.content)
        print()
    loop_invs = extract_loop_invariants(response)
    if verbose_level >= 2:
        print('-'*5, ' Loop Invariants: ', '-'*20)
        print(loop_invs)
        print()

    raw_loop_invs = loop_invs
    check_result, loop_invs = data.check(loop_invs)
    if verbose_level >= 2:
        print('-'*5, ' Check Result: ', '-'*20)
        print(check_result)
        print(loop_invs)
        print()
    if not check_result['success']:
        houdini_check_result, chunked_loop_invs = data.houdini_check(loop_invs)
        success = houdini_check_result['success']
        print('-'*5, ' Houdini Check Result: ', '-'*20)
        print(houdini_check_result)
        print()
        print('-'*5, ' Chunked Loop Invariants: ', '-'*20)
        print(chunked_loop_invs)
        print()
    else:
        success = True
        houdini_check_result = None
        chunked_loop_invs = None

    return {
        'success': success,
        'check_result': check_result,
        'houdini_check_result': houdini_check_result,
        'response': response,
        'raw_loop_invs': raw_loop_invs,
        'loop_invs': loop_invs,
        'chunked_loop_invs': chunked_loop_invs,
        'costs': costs.usage,
    }
