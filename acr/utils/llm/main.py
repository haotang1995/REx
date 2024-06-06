#!/usr/bin/env python
# coding=utf-8

import copy
import hashlib

from openai import OpenAI

from ..caching import _CacheSystem

def add_llm_args(parser):
    parser.add_argument('--llm-model', type=str, default='gpt-4', help="The model to use for LLM")
    parser.add_argument('--llm-temperature', type=float, default=1.0, help="The temperature to use for LLM")
    parser.add_argument('--llm-seed', type=int, default=None, help="The seed to use for LLM")
def get_llm(args):
    if args.llm_seed is None:
        # assert 'seed' in dir(args), f"args: {args}"
        # assert args.seed is not None, f"args: {args}"
        args.llm_seed = args.seed if 'seed' in dir(args) and args.seed is not None else 0
    return LLM(default_args={
        'model': args.llm_model,
        'temperature': args.llm_temperature,
    }, seed=args.llm_seed,)

class LLM:
    def __init__(self, seed=0, default_args={'model': 'gpt-4', 'temperature': 1.0,},):
        self.llm = _LLM(seed=seed, default_args=copy.deepcopy(default_args))
        self.tracker = LLMUsageTracker()
    def __call__(self, prompt, model_args=None):
        completion = self.llm(prompt, model_args)
        self.tracker.update(completion)
        return completion
    def track(self, name=None):
        return self.tracker.track(name)
    @property
    def default_args(self):
        return self.llm.default_args
    def set_seed(self, seed):
        self.llm.set_seed(seed)

class _LLM(_CacheSystem):
    def __init__(self, seed=0, default_args={'model': 'gpt-4', 'temperature': 1.0,},):
        super(_LLM, self).__init__(seed=seed, stochastic=True,)
        self.default_args = default_args
        self.client = OpenAI()
        self.local_tracker = LLMUsageTracker()
    def _action(self, prompt, model_args=None):
        # raise NotImplementedError
        model_args = self._merge_args(model_args)
        # TODO: handle Error and Retry
        out = self.client.chat.completions.create(messages=prompt, **model_args)
        assert str(out).startswith('ChatCompletion('), f"out: {out}"
        self.local_tracker.update(out)
        return out
    def _cache_id(self, prompt, model_args=None):
        assert isinstance(prompt, list), f"prompt: {prompt}"
        assert all(isinstance(p, dict) for p in prompt), f"prompt: {prompt}"
        model_args = self._merge_args(model_args)
        assert isinstance(model_args, dict), f"model_args: {model_args}"
        prompt_id = hashlib.md5(str(prompt).encode()).hexdigest()
        model_args_id = '-'.join([f"{k}_{str(v)[:5]}" for k, v in model_args.items()])
        return (
            ('prompt', prompt_id, prompt),
            ('model_args', model_args_id, model_args),
        )
    def _merge_args(self, model_args):
        args = copy.deepcopy(self.default_args)
        if model_args is not None:
            args.update(model_args)
        return args
    def track(self, name=None):
        return self.local_tracker.track(name)
    def __getstate__(self):
        return {
            'seed': self.seed,
            'default_args': self.default_args,
            'local_tracker': self.local_tracker,
        }
    def __setstate__(self, state):
        self.seed = state['seed']
        self.default_args = state['default_args']
        self.local_tracker = state['local_tracker']

class LLMUsageTracker:
    def __init__(self,):
        self.usage = {
            'requests': 0,
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        }
        self.minor_trackers = {}
    def update(self, completion):
        self.usage['requests'] += 1
        self.usage['completion_tokens'] += completion.usage.completion_tokens
        self.usage['prompt_tokens'] += completion.usage.prompt_tokens
        self.usage['total_tokens'] += completion.usage.total_tokens
    def __str__(self):
        return str(self.usage)
    def __repr__(self):
        return str(self.usage)
    def track(self, name=None):
        if name is None:
            return MinorLLMUsageTracker(self, name)
        if name not in self.minor_trackers:
            self.minor_trackers[name] = MinorLLMUsageTracker(self, name)
        return self.minor_trackers[name]
class MinorLLMUsageTracker():
    def __init__(self, tracker, name):
        self.name = name
        self.tracker = tracker
        self.usage = {
            'requests': 0,
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        }
    def __enter__(self,):
        self.start = copy.deepcopy(self.tracker.usage)
        return self
    def __exit__(self, type, value, traceback):
        self.end = copy.deepcopy(self.tracker.usage)
        self.usage['requests'] += self.end['requests'] - self.start['requests']
        self.usage['completion_tokens'] += self.end['completion_tokens'] - self.start['completion_tokens']
        self.usage['prompt_tokens'] += self.end['prompt_tokens'] - self.start['prompt_tokens']
        self.usage['total_tokens'] += self.end['total_tokens'] - self.start['total_tokens']
    def __str__(self):
        return str(self.usage)
    def __repr__(self):
        return str(self.usage)
