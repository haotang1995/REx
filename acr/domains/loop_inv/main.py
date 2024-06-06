#!/usr/bin/env python
# coding=utf-8

import copy

from ..base import _Domain
from .data.nla_data import add_nla_data_args, get_nla_dataset
from .llm_actions.init import init_loop_inv
from .llm_actions.refine import refine_loop_inv
from ...utils.llm import get_llm
from ...utils.llm.utils import count_tokens_for_openai

def add_loop_inv_args(parser):
    add_nla_data_args(parser)
    parser.add_argument('--houdini-refine', action='store_true', help="Use houdini's result for refinement")
    # parser.add_argument('--houdini-reward', action='store_true', help="Use houdini's result for reward")
    parser.add_argument('--no-houdini-reward', action='store_false', dest='houdini_reward', help="Use houdini's result for reward", default=True)

class InitLoopInv:
    def __init__(self, data, llm):
        self.data = data
        self.llm = llm
        self.name = f"InitLoopInv({data})"
    def run(self):
        return init_loop_inv(self.data, self.llm)
class RefineLoopInv:
    def __init__(self, data, check_result, llm):
        self.data = data
        self.check_result = check_result
        self.llm = llm
        self.name = f"RefineLoopInv({check_result['overall_metrics']['establish']['success_rate']}, {check_result['overall_metrics']['preserve']['success_rate']}, {check_result['overall_metrics']['post']['success_rate']})"
    def run(self):
        return refine_loop_inv(self.data, self.check_result, self.llm)
class LoopInvDomain(_Domain):
    def __init__(
        self, args, verbose=True,
    ):
        self.llm = get_llm(args)
        self.dataset = get_nla_dataset(args)
        self.houdini_refine = args.houdini_refine
        self.houdini_reward = args.houdini_reward
        self.verbose = verbose

    def reset(self, problem_id):
        self.per_request_results = []
        self.problem_id = problem_id
        self.data = self.dataset[problem_id]
        self.actions = [InitLoopInv(self.data, self.llm),]

        self.max_metrics = {
            'success': False,
            'houdini_success': False,
            'merged_success': False,
            'success_in_steps': None,
            'houdini_merged_success': False,
            'merged_success_in_steps': None,
            'pass_rate': 0,
            'houdini_pass_rate': 0,
            'merged_pass_rate': 0,
            'recall': 0,
            'houdini_recall': 0,
            'merged_recall': 0,
            'post_pass_rate': 0,
            'houdini_post_pass_rate': 0,
            'merged_post_pass_rate': 0,
        }
        self.cur_step = 0
        output_new_actions = [
            (i, action.name, None)
            for i, action in enumerate(self.actions)
        ]

        if self.verbose:
            print(f"Problem {problem_id} reset")
        return output_new_actions
    def step(self, action_index):
        assert 0 <= action_index < len(self.actions)
        self.cur_step += 1
        action = self.actions[action_index]
        print()
        print('='*10, f'Step {self.cur_step}', f'Action {action.name}', '='*10)
        result = action.run()
        self.per_request_results.append(result)
        reward, done = self.compute_reward(result)
        heuristic = self.compute_heuristic(result)
        new_actions = self.get_new_actions(result)
        self.update_max_metrics(result)
        self.actions += new_actions
        output_new_actions = [
            (len(self.actions) - len(new_actions) + i, new_action.name, heuristic)
            for i, new_action in enumerate(new_actions)
        ]
        if self.verbose:
            print(f"Step {len(self.per_request_results)}: {action.name} -> {result['success']}, {reward}, {done}")
            print(f"New actions: {output_new_actions}")
        return reward, done, output_new_actions
    def update_max_metrics(self, result):
        # merged_result = self._evaluate_merged_loop_invariants()
        if result['check_result']['success']:
            self.max_metrics['success'] = True
            self.max_metrics['success_in_steps'] = self.cur_step
        if result['check_result']['success'] or result['houdini_check_result']['success']:
            self.max_metrics['houdini_success'] = True
            self.max_metrics['houdini_success_in_steps'] = self.cur_step
        # if merged_result['success']:
            # self.max_metrics['merged_success'] = True
            # self.max_metrics['merged_success_in_steps'] = self.cur_step
        self.max_metrics['pass_rate'] = max(
            self.max_metrics['pass_rate'],
            self._pass_rate(result['check_result']),
        )
        if not result['check_result']['success']:
            self.max_metrics['houdini_pass_rate'] = max(
                self.max_metrics['houdini_pass_rate'],
                self._pass_rate(result['houdini_check_result']),
            )
        else:
            self.max_metrics['houdini_pass_rate'] = 1
        # self.max_metrics['merged_pass_rate'] = max(
            # self.max_metrics['merged_pass_rate'],
            # self._pass_rate(merged_result),
        # )
        self.max_metrics['recall'] = max(
            self.max_metrics['recall'],
            result['check_result']['vs_ground_truth']['recall'],
        )
        if not result['check_result']['success']:
            self.max_metrics['houdini_recall'] = max(
                self.max_metrics['houdini_recall'],
                result['houdini_check_result']['vs_ground_truth']['recall'],
            )
        else:
            self.max_metrics['houdini_recall'] = self.max_metrics['recall']
        # self.max_metrics['merged_recall'] = max(
            # self.max_metrics['merged_recall'],
            # merged_result['vs_ground_truth']['recall'],
        # )
        self.max_metrics['post_pass_rate'] = max(
            self.max_metrics['post_pass_rate'],
            self._post_pass_rate(result['check_result']),
        )
        if not result['check_result']['success']:
            self.max_metrics['houdini_post_pass_rate'] = max(
                self.max_metrics['houdini_post_pass_rate'],
                self._post_pass_rate(result['houdini_check_result']),
            )
        else:
            self.max_metrics['houdini_post_pass_rate'] = self.max_metrics['post_pass_rate']
        # self.max_metrics['merged_post_pass_rate'] = max(
            # self.max_metrics['merged_post_pass_rate'],
            # self._post_pass_rate(merged_result),
        # )
    def get_new_actions(self, result):
        if result['success']:
            return []
        if self.houdini_refine:
            check_result = result['houdini_check_result']
        else:
            check_result = result['check_result']
        if len(check_result['intermediate']['simplified_loop_invariants']) == 0:
            return []
        if set(check_result['intermediate']['simplified_loop_invariants']) == {'True'}:
            return []
        if len(check_result['intermediate']['simplified_loop_invariants']) >= 20:
            return []
        if count_tokens_for_openai('\n'.join(check_result['intermediate']['simplified_loop_invariants'])) >= 500:
            return []
        return [RefineLoopInv(self.data, check_result, self.llm)]
    def _evaluate_merged_loop_invariants(self):
        merged_loop_invariants = [
            inv
            for res in self.per_request_results
            # for inv in res['check_result']['intermediate']['simplified_loop_invariants']
            for inv in res['check_result']['inputs']['loop_invariants']
        ]
        merged_loop_invariants = list(set(merged_loop_invariants))
        # print('merged_loop_invariants:', list(sorted(merged_loop_invariants)), len(merged_loop_invariants))
        merged_res, simplied_merged_loop_invariants = self.data.houdini_check(merged_loop_invariants)
        return merged_res
    def compute_reward(self, result):
        if not self.houdini_reward:
            if result['success']:
                return 1, True
            return 0, False
        else:
            if self.verbose:
                print(f"Use merged_loop_invariants to compute reward")
            merged_res = self._evaluate_merged_loop_invariants()
            print('result:', merged_res)
            if merged_res['success']:
                return 1, True
            return 0, False
    def compute_heuristic(self, result):
        if self.houdini_refine:
            check_result = result['houdini_check_result']
        else:
            check_result = result['check_result']
        return self._pass_rate(check_result)
    def get_metrics(self,):
        return copy.deepcopy(self.max_metrics)
    def _post_pass_rate(self, check_result):
        return check_result['overall_metrics']['post']['success_rate']
    def _pass_rate(self, check_result):
        return (check_result['overall_metrics']['establish']['success_rate'] +
                check_result['overall_metrics']['preserve']['success_rate'] +
                check_result['overall_metrics']['post']['success_rate']) / 3
    def set_seed(self, seed):
        self.llm.set_seed(seed)
    def __len__(self):
        return len(self.dataset)
    def __str__(self,):
        return f"LoopInvDomain({len(self.dataset)})"
