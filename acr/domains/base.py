#!/usr/bin/env python
# coding=utf-8

class _Domain:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    def reset(self, problem_id,):
        # -> list of actions [(action_index, action_name, heuristic_reward)]
        raise NotImplementedError
    def step(self, action):
        # -> reward, done
        # -> list of new actions [(action_index, action_name, heuristic_reward)]
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError
    def __repr__(self):
        return str(self)
    def summarize_results(self, results):
        assert len(results) == len(self)
        raise NotImplementedError
    def set_seed(self, seed):
        raise NotImplementedError
