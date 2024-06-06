import sys
import faulthandler
import platform
# to run the solution files we're using a timing based approach
import signal

import numpy as np
# for capturing the stdout
from io import StringIO
from pyext import RuntimeModule
from unittest.mock import patch, mock_open
import gc
from enum import Enum
from tqdm import tqdm


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("alarm went off")
    # return
    raise TimeoutException


if platform.system() == 'Windows':
    signal.signal(signal.SIGBREAK, timeout_handler)  # windows only
elif platform.system() == 'Linux':
    signal.signal(signal.SIGALRM, timeout_handler)  # linux only


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # TODO: the below line was originally commented
    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', StringIO(inputs))
    @patch('sys.stdin.readline', lambda *args: next(inputs_line_iterator))
    @patch('sys.stdin.readlines', lambda *args: inputs.split("\n"))
    @patch('sys.stdin.read', lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def custom_compare_(output, ground_truth):
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False


def run_test_clean(proposed_solution, test_cases, evaluation_mode, run_all, compile_timeout=4, runtime_timeout=4):
    """
    if test is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    system_platform = platform.system()
    if evaluation_mode == 'standard_input':
        which_type = CODE_TYPE.standard_input  # Standard input
        method_name = None
    elif evaluation_mode == 'function_call':
        which_type = CODE_TYPE.call_based  # Call-based
        method_name = test_cases["fn_name"].strip()
    else:
        raise NotImplementedError
    results = []
    errors = []
    outputs = []
    boilerplate = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
    sol = boilerplate

    if which_type == CODE_TYPE.call_based:
        sol += proposed_solution
        if system_platform == 'Linux':
            signal.alarm(compile_timeout)
        try:
            tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
            if "class Solution" not in proposed_solution:
                tmp = tmp_sol
            else:
                tmp = tmp_sol.Solution()
            if system_platform == 'Linux':
                signal.alarm(0)
        except Exception as e:
            if system_platform == 'Linux':
                signal.alarm(0)
            results.append(-2)
            errors.append(e)
            outputs.append(None)
            return results, errors, outputs, sol
        if system_platform == 'Linux':
            signal.alarm(0)
    elif which_type == CODE_TYPE.standard_input:
        tmp_test = proposed_solution.split("\n")
        # indentation formatting
        new_test = []
        for x in tmp_test:
            if (not x.startswith("from ")) and (not x.startswith("import ")):
                new_test.append("\t" + x + "\n")
            else:
                new_test.append(x + "\n")
        tmp_test = new_test

        new_test = ""
        started = False
        for i in tmp_test:
            if i.startswith("\t") and not started:
                new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                new_test += "def code():\n"
                new_test += i
                started = True
            elif started and (i.startswith("from ") or i.startswith("import ")):
                new_test += "\t" + i
            else:
                new_test += i
        tmp_test = new_test

        sol += tmp_test
        method_name = "code"
        if system_platform == 'Linux':
            signal.alarm(compile_timeout)
        try:
            tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
            tmp = tmp_sol
            if system_platform == 'Linux':
                signal.alarm(0)
        except Exception as e:
            if system_platform == 'Linux':
                signal.alarm(0)
            results.append(-2)
            errors.append(e)
            outputs.append(None)
            return results, errors, outputs, sol
        if system_platform == 'Linux':
            signal.alarm(0)
    else:
        raise NotImplementedError
    try:
        method = getattr(tmp, method_name)  # get_attr second arg must be str
    except Exception as e:  # wrong method name
        assert evaluation_mode == 'function_call'
        results.append(-3)
        errors.append(e)
        outputs.append(None)
        return results, errors, outputs, sol

    for index, inputs in tqdm(enumerate(test_cases["inputs"]), total=len(test_cases["inputs"]), ncols=0, leave=False):
        gc.collect()
        # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
        try:
            if isinstance(inputs[0], dict):
                inputs = [{int(k): v for k, v in inputs[0].items()}]
        except:
            pass
        try:
            if isinstance(test_cases["outputs"][index], dict):
                test_cases["outputs"][index] = [{int(k): v for k, v in test_cases["outputs"][index].items()}]
        except:
            pass
        try:
            if isinstance(test_cases["outputs"][index][0], dict):
                test_cases["outputs"][index] = [{int(k): v for k, v in test_cases["outputs"][index][0].items()}]
        except:
            pass

        if which_type == CODE_TYPE.call_based:  # Call-based
            faulthandler.enable()
            if system_platform == 'Linux':
                signal.alarm(runtime_timeout)
            try:
                output = method(*inputs)

                # ground truth sequences are not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = output == test_cases["outputs"][index]
                if isinstance(test_cases["outputs"][index], list) and test_cases["outputs"][index]:
                    tmp_result = tmp_result or (output == test_cases["outputs"][index][0])

                # ground truth sequences are not tuples
                try:
                    if isinstance(output[0], tuple):
                        tmp_result = tmp_result or ([list(x) for x in output] == test_cases["outputs"][index][0])
                except:
                    pass
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                if system_platform == 'Linux':
                    signal.alarm(0)

            except Exception as e:
                if system_platform == 'Linux':
                    signal.alarm(0)
                faulthandler.disable()
                results.append(-1)
                errors.append(e)
                outputs.append(None)
                if run_all:
                    continue
                else:
                    # TESTING TRICK: exit loop if not pass a test case
                    return results, errors, outputs, sol
            faulthandler.disable()
            if system_platform == 'Linux':
                signal.alarm(0)
            if run_all:
                continue
            if not tmp_result:
                # TESTING TRICK: exit loop if not pass a test case
                return results, errors, outputs, sol

        elif which_type == CODE_TYPE.standard_input:  # Standard input
            faulthandler.enable()

            if isinstance(inputs, list):
                inputs = "\n".join(inputs)
            if isinstance(test_cases['outputs'][index], list):
                test_cases['outputs'][index] = "\n".join(test_cases['outputs'][index])
            if system_platform == 'Linux':
                signal.alarm(runtime_timeout)
            with Capturing() as output:
                try:
                    call_method(method, inputs)
                    if system_platform == 'Linux':
                        signal.alarm(0)
                except Exception as e:
                    if system_platform == 'Linux':
                        signal.alarm(0)
                    results.append(-1)
                    errors.append(e)
                    outputs.append(None)
                    if run_all:
                        continue
                    else:
                        # TESTING TRICK: exit loop if not pass a test case
                        return results, errors, outputs, sol
                if system_platform == 'Linux':
                    signal.alarm(0)
            original_output = output.copy()
            if custom_compare_(output, test_cases['outputs'][index]):
                tmp_result = True
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                continue

            # ground truth sequences are expressed as lists not tuples
            if isinstance(output, tuple):
                output = list(output)

            tmp_result = False
            try:
                tmp_result = (output == [test_cases["outputs"][index]])
                if isinstance(test_cases["outputs"][index], list):
                    tmp_result = tmp_result or (output == test_cases["outputs"][index])
                    if isinstance(output[0], str):
                        tmp_result = tmp_result or ([e.strip() for e in output] == test_cases["outputs"][index])
            except Exception as e:
                pass

            if tmp_result:
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                continue

            # try one more time without \n
            if isinstance(test_cases["outputs"][index], list):
                for tmp_index, i in enumerate(test_cases["outputs"][index]):
                    test_cases["outputs"][index][tmp_index] = i.split("\n")
                    test_cases["outputs"][index][tmp_index] = [x.strip() for x in
                                                               test_cases["outputs"][index][tmp_index] if x]
            else:
                test_cases["outputs"][index] = test_cases["outputs"][index].split("\n")
                test_cases["outputs"][index] = list(filter(len, test_cases["outputs"][index]))
                test_cases["outputs"][index] = list(map(lambda x: x.strip(), test_cases["outputs"][index]))

            try:
                tmp_result = (output == [test_cases["outputs"][index]])
                if isinstance(test_cases["outputs"][index], list):
                    tmp_result = tmp_result or (output == test_cases["outputs"][index])
            except Exception as e:
                pass

            if tmp_result:
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                continue

            # try by converting the output into a split up list too
            if isinstance(output, list):
                output = list(filter(len, output))

            try:
                tmp_result = (output == [test_cases["outputs"][index]])
                if isinstance(test_cases["outputs"][index], list):
                    tmp_result = tmp_result or (output == test_cases["outputs"][index])
            except Exception as e:
                pass

            if tmp_result:
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                continue

            try:
                output_float = [float(e) for e in output]
                gt_float = [float(e) for e in test_cases['outputs'][index]]
                tmp_result = tmp_result or (
                        (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
            except Exception as e:
                pass

            if tmp_result:
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output_float)
                continue

            try:
                if isinstance(output[0], list):
                    output_float = [float(e) for e in output[0]]
                    gt_float = [float(e) for e in test_cases['outputs'][index][0]]
                    tmp_result = tmp_result or (
                            (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
            except Exception as e:
                pass

            if tmp_result:
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output_float)
                continue

            # try by converting the stuff into split up list
            if isinstance(test_cases["outputs"][index], list):
                for tmp_index, i in enumerate(test_cases["outputs"][index]):
                    test_cases["outputs"][index][tmp_index] = set(i.split())
            else:
                test_cases["outputs"][index] = set(test_cases["outputs"][index].split())

            try:
                tmp_result = (output == test_cases["outputs"][index])
            except Exception as e:
                pass

            if tmp_result:
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                continue

            # try by converting the output into a split up list too
            if isinstance(output, list):
                for tmp_index, i in enumerate(output):
                    output[tmp_index] = i.split()
                output = list(filter(len, output))
                for tmp_index, i in enumerate(output):
                    output[tmp_index] = set(i)
            else:
                output = output.split()
                output = list(filter(len, output))
                output = set(output)

            try:
                tmp_result = (set(frozenset(s) for s in output) == set(
                    frozenset(s) for s in test_cases["outputs"][index]))
            except Exception as e:
                pass

            if tmp_result:
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                continue

            # if they are all numbers, round so that similar numbers are treated as identical
            try:
                tmp_result = tmp_result or (set(frozenset(round(float(t), 3) for t in s) for s in output) == set(
                    frozenset(round(float(t), 3) for t in s) for s in test_cases["outputs"][index]))
            except Exception as e:
                pass

            if tmp_result:
                results.append(tmp_result)
                errors.append(None)
                outputs.append(output)
                continue

            results.append(tmp_result)
            errors.append(None)
            outputs.append(original_output)  # failure append original output for easier analysis
            if run_all:
                continue
            if not tmp_result:
                # TESTING TRICK: exit loop if not pass a test case
                return results, errors, outputs, sol

    return results, errors, outputs, sol
