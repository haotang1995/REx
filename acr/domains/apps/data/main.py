#!/usr/bin/env python
# coding=utf-8

import os
import json
import hashlib

from ..from_jin.utility import read_problems_data as _read_problems_data
from ..from_jin.utility import run_single_solution_test as _run_single_solution_test
from ....utils.caching import _CacheSystem

def add_apps_data_args(parser):
    parser.add_argument(
        '--apps-difficulty', type=str, default='intro',
        choices=['intro', 'interview', 'comp'],
        help='APPS difficulty level',
    )
def get_apps_dataset(args):
    return APPSDataset(args.apps_difficulty)

class APPSDataset:
    def __init__(self, difficulty,):
        self.difficulty = difficulty
        self.cached_check = APPSCachedCheck()
        curdir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(curdir, 'benchmarks', 'test-problem-ids-%s.txt' % self.difficulty)) as f:
            self.problem_ids = f.read().splitlines()
        self.data_path = os.path.join(curdir, 'benchmarks', 'APPS', 'test') + '/'
        self.problems = [
            APPSProblem(
                problem_id,
                self.data_path,
                cached_check=self.cached_check,
            )
            for problem_id in self.problem_ids
        ]
    def __len__(self):
        return len(self.problems)
    def __getitem__(self, idx):
        return self.problems[idx]
    def __iter__(self):
        return iter(self.problems)
    def __repr__(self):
        return 'APPSDataset(%s, %s)' % (self.difficulty, len(self))
    def __str__(self):
        return str([str(problem) for problem in self])

class APPSProblem:
    def __init__(
        self, problem_id, data_path,
        cached_check=None,
    ):
        self.problem_id = problem_id
        self.data_path = data_path
        self.data = _read_problems_data(self.data_path, [str(self.problem_id)])[0]
        with open(os.path.join(self.data_path, str(self.problem_id), 'input_output.json')) as f:
            self.test_cases = json.load(f)
        if self.data['has_solution']:
            with open(os.path.join(self.data_path, str(self.problem_id), 'solutions.json')) as f:
                self.solutions = json.load(f)
        else:
            self.solutions = []
        self.cached_check = cached_check
    def __repr__(self):
        return 'APPSProblem(%s)' % self.problem_id
    def __str__(self):
        return self.__repr__()
    def check(self, solution):
        return self.cached_check(self.test_cases, solution)
    def __getitem__(self, key):
        return self.data[key]

class APPSCachedCheck(_CacheSystem):
    def __init__(self, compile_timeout=20, runtime_timeout=5,):
        assert compile_timeout == 20 and runtime_timeout == 5, 'APPS does not support custom timeouts'
        super().__init__(None, seed=0, stochastic=False)
        self.compile_timeout = compile_timeout
        self.runtime_timeout = runtime_timeout
    def _action(self, test_cases, solution,):
        if solution is None:
            return {
                'message': 'No solution provided',
                'success': False,
                'total': len(test_cases['inputs']),
                'succeeded': 0,
                'failed': len(test_cases['inputs']),
                'success_rate': 0.0,
                'solution': None,
            }
        assert solution is not None, solution
        all_results = dict()
        try:
            message = _run_single_solution_test(
                test_cases, solution, False, True,
                self.compile_timeout, self.runtime_timeout,
            )
        except Exception as e:
            print('Failed solution {0} tests results: {1}'.format(solution, e))
            assert 0 == 1  # temporary solution
        all_results['evaluation'] = message
        all_results['solution'] = solution  # should already be extracted anyway
        all_results['success'] = (all_results['evaluation'] == 'all test cases passed')
        all_results['total'] = len(test_cases['inputs'])
        if all_results['success']:
            all_results['success_rate'] = 1.0
            all_results['succeeded'] = all_results['total']
            all_results['failed'] = 0
        else:
            assert 'Overall evaluation:' in message, message
            overall_message = message.split('Overall evaluation:')[1].strip()
            succeeded = int(overall_message.split(' ')[0])
            try:
                total = int(overall_message[overall_message.find('out of') + 6:].strip().split(' ')[0])
            except Exception as e:
                print(message)
                print('='*80)
                print(overall_message)
                raise e
            assert total == all_results['total'], (total, all_results['total'], message)
            all_results['succeeded'] = succeeded
            all_results['failed'] = total - succeeded
            all_results['success_rate'] = succeeded / total
        return all_results
    def _cache_id(self, test_cases, solution):
        assert isinstance(test_cases, dict)
        assert set(['inputs', 'outputs']).issubset(set(test_cases.keys())), test_cases
        assert isinstance(test_cases['inputs'], list)
        assert isinstance(test_cases['outputs'], list)
        assert isinstance(solution, str) or solution is None, solution

        test_cases_id = hashlib.md5(str(list(sorted(list(test_cases.items()), key=str))).encode()).hexdigest()
        solution_id = hashlib.md5(str(solution).encode()).hexdigest()
        return (
            ('test_cases', test_cases_id, test_cases),
            ('solution', solution_id, solution),
        )
