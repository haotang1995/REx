#!/usr/bin/env python
# coding=utf-8

import copy

from ....utils.llm import LLM
from ..from_jin.openai_utils import construct_iterative_prompts as _construct_iterative_prompts
from ..from_jin.utility import extract_actual_solution as _extract_actual_solution
from .utils import _message_d

def refine_apps(data, check_result, llm, verbose_level=2):
    assert isinstance(llm, LLM)
    assert not check_result['success']

    request_data = copy.deepcopy(data.data)
    request_data['evaluation_message'] = check_result['evaluation']
    request_data['evaluation_solution'] = check_result['solution']
    assert request_data['evaluation_message'] != 'all test cases passed'
    request_data = _construct_iterative_prompts([request_data])[0]
    prompt = copy.deepcopy(_message_d)
    prompt[-1]['content'] = request_data['prompt']
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

    solution = _extract_actual_solution({request_data['question_name']: request_data}, request_data['question_name'], response.choices[0].message.content,)
    if verbose_level >= 2:
        print('-'*5, ' Extracted Solution: ', '-'*20)
        print(solution)
        print()
    check_result = data.check(solution)
    if verbose_level >= 2:
        print('-'*5, ' Check Result: ', '-'*20)
        print(check_result)
        print()
    success = check_result['success']

    return {
        'success': success,
        'check_result': check_result,
        'response': response,
        'costs': costs.usage,
    }
