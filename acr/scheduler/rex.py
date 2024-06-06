#!/usr/bin/env python
# coding=utf-8

import numpy as np

def add_rex_args(parser):
    parser.add_argument('--rex-smoothing', type=float, default=1)
    parser.add_argument('--rex-constant', type=float, default=1)
def get_rex_args(args):
    return {
        'smoothing': args.rex_smoothing,
        'constant': args.rex_constant,
    }

def rex(
    domain, problem_id, seed,
    smoothing=1, constant=5, max_steps=300,
):
    rng = np.random.default_rng(seed)
    domain.set_seed(seed)
    def new_action(act):
        action_index, action_name, heuristic_reward = act
        return {
            'index': action_index,
            'name': action_name,
            'heuristic_reward': heuristic_reward,
            'alpha': smoothing if heuristic_reward is None else smoothing + constant * heuristic_reward,
            'beta': smoothing if heuristic_reward is None else smoothing + constant * (1 - heuristic_reward),
        }
    actions = [new_action(act) for act in domain.reset(problem_id)]

    metrics = []
    for si in range(max_steps):
        action = max(actions, key=lambda a: rng.beta(a['alpha'], a['beta']))
        reward, done, new_actions = domain.step(action['index'])
        metrics.append(domain.get_metrics())
        if done:
            break
        action['alpha'] += reward
        action['beta'] += 1 - reward
        actions.extend([new_action(act) for act in new_actions])

    metrics += [metrics[-1]] * (max_steps - len(metrics))

    return metrics
