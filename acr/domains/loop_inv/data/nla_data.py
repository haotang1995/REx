#!/usr/bin/env python
# coding=utf-8

import z3
import os
import hashlib

from .from_gcln._trace_related import get_traces
from .from_gcln.run_nla import PROBLEMS
from .z3_checks import check_z3_problem, compare_with_ground_truth
from ....utils.caching import _CacheSystem

def add_nla_data_args(parser):
    parser.add_argument("--with-post-cond", action="store_true", help="Whether to include post-condition in the loop invariant")
def get_nla_dataset(args):
    return NLADataset(with_post_cond=args.with_post_cond)

curdir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(curdir, "benchmarks")
nla_path = os.path.join(data_path, "nla")
nla_code_path = os.path.join(nla_path, "c")
nla_trace_path = os.path.join(nla_path, "csv")

class NLADataset:
    def __init__(self, with_post_cond=False):
        self.cached_check = CachedCheck()
        self.cached_houdini_check = CachedHoudiniCheck(cached_check=self.cached_check)
        self.problems = [
            NLADataPerLoop(
                problem.name,
                loop_index=problem.loop_index,
                with_post_cond=with_post_cond,
                cached_check=self.cached_check,
                cached_houdini_check=self.cached_houdini_check,
            )
            for problem_set in PROBLEMS.values()
            for problem in problem_set
        ]
    def __getitem__(self, idx):
        return self.problems[idx]
    def __len__(self,):
        return len(self.problems)
    def __iter__(self,):
        return iter(self.problems)
    def __repr__(self,):
        return f"NLADataset({len(self.problems)} problems)"
    def __str__(self,):
        return str(self.problems)
    def cal_metrics(self, loop_invariants):
        assert len(loop_invariants) == len(self.problems)
        check_results = []
        for loop_invariant, problem in zip(loop_invariants, self.problems):
            check_res = problem.check(loop_invariant)
            check_results.append(check_res)
        raise NotImplementedError

class NLADataPerLoop:
    def __init__(self, name, loop_index=1, with_post_cond=False,
                 cached_check=None, cached_houdini_check=None):
        self.name = name
        self.loop_index = loop_index
        self.with_post_cond = with_post_cond
        self.cached_check = cached_check
        self.cached_houdini_check = cached_houdini_check
        assert self.cached_check is not None
        assert self.cached_houdini_check is not None
        self._code = None
        self._traces = None
    def __repr__(self,):
        return f"NLADataPerLoop(name={self.name}, loop_index={self.loop_index})"
    def __str__(self,):
        return self.__repr__()
    def check(self, loop_invariants):
        return self.cached_check(self.name, self.loop_index, loop_invariants, self.true_loop_invariants)
    def houdini_check(self, loop_invariants):
        return self.cached_houdini_check(self.name, self.loop_index, loop_invariants, self.true_loop_invariants)
    @property
    def code(self):
        if self._code is None:
            with open(os.path.join(nla_code_path, f"{self.name}.c"), "r") as f:
                self._code = f.read()
        return self._code
    @property
    def traces(self):
        if self._traces is None:
            self._traces = get_traces(self.name, self.loop_index, nla_trace_path)
        return self._traces
    @property
    def true_loop_invariants(self,):
        code = self.code
        num_while = code.count("while")
        assert self.loop_index <= num_while
        for _ in range(self.loop_index):
            code = code[code.find("while")+5:]
        if 'while' in code:
            code = code[:code.find("while")]
        else:
            code = code[:code.find("}")]
        lines = [l.strip() for l in code.split("\n")]
        loop_invariants = []
        for l in lines:
            if l.startswith("//assert"):
                l = l[len("//assert"):].strip().strip(';').strip()
                if l.startswith('(') and l.endswith(')'):
                    l = l[1:-1]
                for ll in l.split("&&"):
                    loop_invariants.append(ll.strip())
        return loop_invariants
    def code_for_llm(self):
        code = self.code
        lines = code.split("\n")
        # Remove #include <assert.h>
        lines = [l for l in lines if not l.lstrip().startswith("#include <assert.h>")]
        # //* -> NULL (still keep //assert*)
        lines = [l for l in lines if not l.lstrip().startswith("//") or l.lstrip().startswith("//assert")]
        # /* * */ -> NULL
        lines = [l for l in lines if not l.lstrip().startswith("/*") and not l.rstrip().endswith("*/")]
        # Remove int main() { ... }
        index_main = next(i for i, l in enumerate(lines) if l.lstrip().startswith("int main("))
        lines = lines[:index_main]
        code = "\n".join(lines)
        # Rename mainQ to main
        code = code.replace("mainQ(", "main(")

        # //assert* -> //assert
        start_of_loop = 0
        for _ in range(self.loop_index):
            start_of_loop = code.find("while", start_of_loop) + 5
        if 'while' in code[start_of_loop:]:
            end_of_loop = code.find("while", start_of_loop)
        else:
            end_of_loop = code.find("}", start_of_loop)

        code_before_loop = code[:start_of_loop-5]
        code_in_loop = code[start_of_loop-5:end_of_loop]
        code_after_loop = code[end_of_loop:]

        end_of_all_loop = code_after_loop[:code_after_loop.rfind("}")].rfind("}")
        code_after_butinanotherloop = code_after_loop[:end_of_all_loop+1]
        code_after_after_loop = code_after_loop[end_of_all_loop+1:]

        code_before_loop = self._remove_comments(code_before_loop)
        code_after_butinanotherloop = self._remove_comments(code_after_butinanotherloop)
        if not self.with_post_cond:
            code_after_after_loop = self._remove_comments(code_after_after_loop)
        code_after_loop = code_after_butinanotherloop + '\n' + code_after_after_loop

        lines = code_in_loop.split("\n")
        # Replace //assert* with // loop invariant in Z3: in loop
        already_replaced = False
        index_to_remove = []
        for i, l in enumerate(lines):
            if l.lstrip().startswith("//assert"):
                if not already_replaced:
                    lines[i] = l[:l.find("//assert")] + "// loop invariant in Z3:"
                    already_replaced = True
                else:
                    lines[i] = ''
                    index_to_remove.append(i)
        lines = [l for i, l in enumerate(lines) if i not in index_to_remove]
        code_in_loop = "\n".join(lines)

        return code_before_loop + code_in_loop + code_after_loop
    @staticmethod
    def _remove_comments(code):
        lines = code.split("\n")
        # //* -> NULL (still keep //assert*)
        lines = [l for l in lines if not l.lstrip().startswith("//")]
        return "\n".join(lines)

class CachedCheck(_CacheSystem):
# class CachedTimedCheck(_CacheSystem):
# class CachedTimedCheck2(_CacheSystem):
    def __init__(self, cache_dir=None):
        super().__init__(cache_dir, seed=0, stochastic=False)
    def _action(self, name, loop_index, loop_invariants, true_loop_invariants):
        assert isinstance(loop_invariants, (list, tuple))
        assert all(isinstance(x, str) for x in loop_invariants)
        check_res, simplified_loop_invariants = check_z3_problem(name, loop_index, loop_invariants)
        check_res['vs_ground_truth'] = compare_with_ground_truth(simplified_loop_invariants, true_loop_invariants)
        if check_res['vs_ground_truth']['recall'] >= 1:
            if not check_res['success']:
                if check_res['overall_metrics']['post']['overall']['_checker_result'] == z3.unknown and check_res['overall_metrics']['inductive']['success']:
                    check_res['success'] = True
                    check_res['overall_metrics']['post']['overall']['success'] = True
                    check_res['overall_metrics']['post']['overall']['success_rate'] = 1.
        return check_res, simplified_loop_invariants
    def _cache_id(self, name, loop_index, loop_invariants, true_loop_invariants):
        name_id = hashlib.md5(name.encode()).hexdigest()
        loop_index_id = hashlib.md5(str(loop_index).encode()).hexdigest()
        loop_invariants_id = hashlib.md5(str(sorted(set(loop_invariants))).encode()).hexdigest()
        true_loop_invariants_id = hashlib.md5(str(sorted(set(true_loop_invariants))).encode()).hexdigest()
        return (
            ('name', name_id, name),
            ('loop_index', loop_index_id, loop_index),
            ('loop_invariants', loop_invariants_id, loop_invariants),
            ('true_loop_invariants', true_loop_invariants_id, true_loop_invariants),
        )
# CachedCheck = CachedTimedCheck2

class CachedHoudiniCheck(_CacheSystem):
# class CachedTimedHoudiniCheck(_CacheSystem):
# class CachedTimedHoudiniCheck2(_CacheSystem):
    def __init__(self, cache_dir=None, cached_check=None):
        super().__init__(cache_dir, seed=0, stochastic=False)
        self.cached_check = cached_check
    def _action(self, name, loop_index, loop_invariants, true_loop_invariants):
        loop_invariants = list(set(loop_invariants))
        while len(loop_invariants) >= 0:
            check_res, loop_invariants = self.cached_check(name, loop_index, loop_invariants, true_loop_invariants)
            if check_res['success']:
                break;
            if len(loop_invariants) == 0:
                break;
            if not check_res['overall_metrics']['establish']['success']: # Remove all loop-inv that are not estabilished
                loop_invariants = [inv for inv, res in zip(loop_invariants, check_res['per_inv_metrics']) if res['establish']['success']]
            elif not check_res['overall_metrics']['preserve']['success']: # Remove the first loop-inv that are not preserved
                index = next(i for i, res in enumerate(check_res['per_inv_metrics']) if not res['preserve']['success'])
                loop_invariants.pop(index)
            else:
                break;
        if check_res['vs_ground_truth']['recall'] >= 1:
            assert check_res['success'], f"Recall >= 1 but success is False: {check_res}"
        return check_res, loop_invariants
    def _cache_id(self, name, loop_index, loop_invariants, true_loop_invariants):
        name_id = hashlib.md5(name.encode()).hexdigest()
        loop_index_id = hashlib.md5(str(loop_index).encode()).hexdigest()
        loop_invariants_id = hashlib.md5(str(sorted(set(loop_invariants))).encode()).hexdigest()
        true_loop_invariants_id = hashlib.md5(str(sorted(set(true_loop_invariants))).encode()).hexdigest()
        return (
            ('name', name_id, name),
            ('loop_index', loop_index_id, loop_index),
            ('loop_invariants', loop_invariants_id, loop_invariants),
            ('true_loop_invariants', true_loop_invariants_id, true_loop_invariants),
        )
# CachedHoudiniCheck = CachedTimedHoudiniCheck2
