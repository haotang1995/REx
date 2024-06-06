import os
import json
from .utils.testing_util import run_test_clean
import traceback
import numpy as np
import copy


def read_jsonl(path):
    results = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        results.append(json.loads(line))
    return results


def extract_actual_solution(problem_data, problem_name, solution):
    starter_code = problem_data[problem_name]['starter_code']
    starter_code = starter_code.rstrip()
    if '```python' in solution:
        start_index = solution.index('```python') + len('```python')
    elif '``` python' in solution:
        start_index = solution.index('``` python') + len('``` python')
    elif '```' in solution:
        start_index = solution.index('```') + len('```')
    else:
        return None
    return solution[start_index:].lstrip().rsplit('```', 1)[0]


def read_problems_data(data_path, problem_names):
    results = []
    for e in problem_names:
        current_d = {'question_name': e}
        question_path = data_path + e + '/question.txt'
        test_cases_path = data_path + e + '/input_output.json'
        starter_code_path = data_path + e + '/starter_code.py'
        solution_path = data_path + e + '/solutions.json'
        if os.path.exists(solution_path):
            current_d['has_solution'] = 1
        else:
            current_d['has_solution'] = 0
        with open(question_path, 'r') as f:
            question_text = f.read()
        with open(test_cases_path, 'r') as f:
            test_cases = json.load(f)
        if 'fn_name' in test_cases:
            evaluate_mode_string = 'function_call'
            assert os.path.exists(starter_code_path)
            with open(starter_code_path, 'r') as f:
                starter_code = f.read()
        else:
            evaluate_mode_string = "standard_input"
            starter_code = ''
        current_d['question'] = question_text
        current_d['evaluate_mode_string'] = evaluate_mode_string
        current_d['starter_code'] = starter_code
        results.append(current_d)
    return results


def convert_jsonl_to_d(problem_data, key='question_name'):
    d = {}
    for i in range(len(problem_data)):
        current_k = problem_data[i][key]
        assert current_k not in d
        d[current_k] = problem_data[i]
    return d


def create_dir_if_necessary(file_path):
    file_dir = file_path.rsplit('/', 1)[0]
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def write_jsonl(results, path):
    create_dir_if_necessary(path)
    with open(path, 'w') as f:
        f.write('\n'.join(json.dumps(e) for e in results))


def get_depth(evaluation_path):
    s = evaluation_path.rsplit('/', 1)[-1].replace('.jsonl', '')
    assert 't' not in s
    return s.count('-')


def analyze_error(tests, line_number_offset, curr_results, curr_errors, outputs, curr_sol, run_all, truncate_actual_output_limit=500):
    first_result = None
    first_result_index = None
    for e in curr_results:
        if e != True:
            first_result = e
            first_result_index = curr_results.index(e)
            break
    num_test_cases_passed = len([e for e in curr_results if e == True])
    if first_result is None:
        message = 'all test cases passed'
    elif first_result == -3:
        assert len(curr_results) == 1
        assert num_test_cases_passed == 0
        error_type = type(curr_errors[first_result_index][0]).__name__
        error_message = curr_errors[first_result_index][0].__str__()
        assert error_message.startswith("module 'tmp_sol' has no attribute ")
        error_message = error_message.replace("module 'tmp_sol' has no attribute", 'proposed solution does not have the correct starter code signature')
        message = 'starter code signature mismatch error. Error type: {0}, detailed error message: {1}'.format(error_type, error_message)
    elif first_result == -2:
        assert len(curr_results) == 1
        assert num_test_cases_passed == 0
        error_type = type(curr_errors[first_result_index][0]).__name__
        if error_type == 'TimeoutException':
            raise NotImplementedError
        elif error_type in ['SyntaxError', 'IndentationError']:
            error_message = curr_errors[first_result_index][0].msg
            raw_line_number = curr_errors[first_result_index][0].lineno
            actual_line_number = raw_line_number - line_number_offset
            code_segment = curr_sol.split('\n')[raw_line_number - 1]
            if run_all:
                message = 'compilation error. Error type: {0}, detailed error message: {1} at line {2}, {3}\nOverall evaluation: 0 out of {4} test cases passed'.format(
                    error_type, error_message, actual_line_number, code_segment.strip(), len(tests['inputs']))

            else:
                message = 'compilation error. Error type: {0}, detailed error message: {1} at line {2}, {3}'.format(error_type, error_message, actual_line_number, code_segment.strip())
        else:
            error_message = curr_errors[first_result_index][0].__str__()
            actual_error_location = []
            for e in curr_errors[first_result_index][1]:
                if e.strip().startswith('File "<string>", line '):
                    actual_error_location.append(e)
            assert len(actual_error_location) >= 1  # hopefully this is the case
            latest_trace = actual_error_location[-1].split(',')  # latest trace is the rightmost one
            assert latest_trace[1].startswith(' line ')
            raw_line_number = int(latest_trace[1].strip().replace('line ', ''))
            actual_line_number = raw_line_number - line_number_offset
            code_segment = curr_sol.split('\n')[raw_line_number - 1]
            if run_all:
                message = 'compilation error. Error type: {0}, detailed error message: {1} at line {2}, {3}\nOverall evaluation: 0 out of {4} test cases passed'.format(
                    error_type, error_message, actual_line_number, code_segment.strip(), len(tests['inputs']))
            else:
                message = 'compilation error. Error type: {0}, detailed error message: {1} at line {2}, {3}'.format(error_type, error_message, actual_line_number, code_segment.strip())
    elif first_result == -1:
        error_type = type(curr_errors[first_result_index][0]).__name__
        if type(tests['inputs'][first_result_index]) == str:
            input_formatted = tests['inputs'][first_result_index].replace('\n', '\\n')
        else:
            input_formatted = tests['inputs'][first_result_index]
        if type(tests['outputs'][first_result_index]) == str:
            output_formatted = tests['outputs'][first_result_index].replace('\n', '\\n')
        else:
            output_formatted = tests['outputs'][first_result_index]
        if error_type == 'TimeoutException':
            if run_all:
                message = 'runtime error at test case {0} for input {1} and expected output {2}. Error type: {3}\nOverall evaluation: {4} out of {5} test cases passed'.format(first_result_index, input_formatted, output_formatted, error_type, num_test_cases_passed, len(tests['inputs']))
            else:
                message = 'runtime error at test case {0} for input {1} and expected output {2}. Error type: {3}'.format(first_result_index, input_formatted, output_formatted, error_type)
        else:
            error_message = curr_errors[first_result_index][0].__str__()
            actual_error_location = []

            # handle number of arguments mismatch for starter code
            if len(curr_errors[first_result_index][1]) == 1 and curr_errors[first_result_index][1][0].endswith('in run_test_clean\n    output = method(*inputs)\n'):
                message = 'starter code signature mismatch error. Error type: {0}, detailed error message: {1}'.format(error_type, error_message)
                return message
            for e in curr_errors[first_result_index][1]:
                if e.strip().startswith('File "<string>", line '):
                    actual_error_location.append(e)
            assert len(actual_error_location) >= 1  # hopefully this is the case
            latest_trace = actual_error_location[-1].split(',')  # latest trace is the rightmost one
            assert latest_trace[1].startswith(' line ')
            raw_line_number = int(latest_trace[1].strip().replace('line ', ''))
            actual_line_number = raw_line_number - line_number_offset
            code_segment = curr_sol.split('\n')[raw_line_number - 1]
            if run_all:
                message = 'runtime error at test case {0} for input {1} and expected output {2}. Error type: {3}, detailed ' \
                          'error message: {4} at line {5}, {6}\nOverall evaluation: {7} out of {8} test cases passed'.format(first_result_index, input_formatted,
                                                                       output_formatted, error_type, error_message,
                                                                       actual_line_number, code_segment.strip(), num_test_cases_passed, len(tests['inputs']))
            else:
                message = 'runtime error at test case {0} for input {1} and expected output {2}. Error type: {3}, detailed ' \
                      'error message: {4} at line {5}, {6}'.format(first_result_index, input_formatted, output_formatted, error_type, error_message, actual_line_number, code_segment.strip())
    elif not first_result:
        actual_output = outputs[first_result_index]
        if actual_output == []:
            formatted_output = '<non-existent>'
        elif type(actual_output) == list and all([type(e) == str for e in actual_output]):
            formatted_output = '\\n'.join(actual_output) + '\\n'
        else:
            formatted_output = actual_output
        if type(tests['inputs'][first_result_index]) == str:
            input_formatted = tests['inputs'][first_result_index].replace('\n', '\\n')
        else:
            input_formatted = tests['inputs'][first_result_index]
        if type(tests['outputs'][first_result_index]) == str:
            output_formatted = tests['outputs'][first_result_index].replace('\n', '\\n')
        else:
            output_formatted = tests['outputs'][first_result_index]
        if truncate_actual_output_limit != -1:
            if len(str(formatted_output)) > truncate_actual_output_limit:
                truncated_output = str(formatted_output)[:truncate_actual_output_limit] + '... (output truncated)'
                print('truncating evaluation actual output to {0} characters'.format(truncate_actual_output_limit))
            else:
                truncated_output = formatted_output
        else:
            truncated_output = formatted_output
        if truncated_output == '<non-existent>':
            message = 'test case failed at test case {0} for input {1} and expected output {2}. The code did not output anything'.format(
                first_result_index, input_formatted, output_formatted)
        else:
            message = 'test case failed at test case {0} for input {1} and expected output {2}, actual output is {3}'.format(
                first_result_index, input_formatted, output_formatted, truncated_output)
        if run_all:
            message += '\nOverall evaluation: ' + str(num_test_cases_passed) + ' out of ' + str(len(tests['inputs'])) + ' test cases passed'
    else:
        raise NotImplementedError
    return message


def run_single_solution_test(tests, proposed_solution, use_old_run_func, run_all_tests, compile_timeout, runtime_timeout):
    if 'fn_name' in tests:
        evaluation_mode = 'function_call'
        line_number_offset = 15  # hard coded
    else:
        evaluation_mode = 'standard_input'
        line_number_offset = 18  # hard coded
    curr_results = []
    tests_copy = copy.deepcopy(tests)  # to make test cases unchanged when analyzing error
    try:
        if use_old_run_func:
            raise NotImplementedError
        else:
            curr_results, curr_errors, outputs, curr_sol = run_test_clean(proposed_solution, tests_copy, evaluation_mode, run_all_tests, compile_timeout=compile_timeout, runtime_timeout=runtime_timeout)
        curr_errors = [(e, traceback.format_tb(e.__traceback__)) if e is not None else e for e in curr_errors]
        fixed = []
        for e in curr_results:
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        curr_results = fixed
    except Exception as e:
        print(f"test framework exception = {repr(e)}{e}\n")
        assert 0 == 1
    finally:
        assert isinstance(curr_results, list)
    message = analyze_error(tests, line_number_offset, curr_results, curr_errors, outputs, curr_sol, run_all_tests)
    return message
