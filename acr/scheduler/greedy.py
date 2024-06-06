#!/usr/bin/env python
# coding=utf-8

import numpy as np

def add_greedy_args(parser):
    parser.add_argument('--greedy-init-value', type=float, default=0.5)
def get_greedy_args(args):
    return {
        'init_value': args.greedy_init_value,
    }

def greedy(
    domain, problem_id, seed,
    init_value=0.5, max_steps=300,
):
    rng = np.random.default_rng(seed)
    domain.set_seed(seed)
    def new_action(act):
        action_index, action_name, heuristic_reward = act
        return {
            'index': action_index,
            'name': action_name,
            'heuristic_reward': heuristic_reward,
            'value': heuristic_reward if heuristic_reward is not None else init_value,
        }
    actions = [new_action(act) for act in domain.reset(problem_id)]

    metrics = []
    for si in range(max_steps):
        action = max(actions, key=lambda a: a['value'])
        reward, done, new_actions = domain.step(action['index'])
        metrics.append(domain.get_metrics())
        if done:
            break
        actions.extend([new_action(act) for act in new_actions])

    metrics += [metrics[-1]] * (max_steps - len(metrics))

    return metrics
