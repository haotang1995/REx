#!/usr/bin/env python
# coding=utf-8

from ....utils.llm import LLM
from .utils import extract_loop_invariants

def init_loop_inv(data, llm, verbose_level=2):
    assert isinstance(llm, LLM)

    code = data.code_for_llm()
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


Please analyze the loop variant in your own language. The loop invariant should be a condition that must be true before and after each iteration of the loop. The loop invariant should help you understand the behavior of the loop and ensure that the loop is working correctly. The loop invariant should also be strong enough to prove the correctness of the loop as well as the functionality of the whole code snippet.

What are the loop invariants in code? The loop invariant should be written in the format as Python-Z3 such as follows:


```

And(x >= 0, x <= 10, Implies(x >= 0, a == 0), y == x * x)

```
Please surround the loop invariant with triple backticks and write it in the format as Python-Z3.
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
