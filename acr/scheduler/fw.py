#!/usr/bin/env python
# coding=utf-8

import numpy as np

def add_fw_args(parser):
    parser.add_argument('--fw-width', type=int, default=100)
    parser.add_argument('--fw-depth', type=int, default=3)
def get_fw_args(args):
    return {
        'width': args.fw_width,
        'depth': args.fw_depth,
    }

def fw(
    domain, problem_id, seed,
    width=100, depth=1, max_steps=300,
):
    # rng = np.random.default_rng(seed)
    domain.set_seed(seed)

    actions = domain.reset(problem_id)
    assert len(actions) == 1

    metrics = []
    init_action = actions[0]
    for si in range(min(max_steps, width)):
        reward, done, new_actions = domain.step(init_action[0])
        metrics.append(domain.get_metrics())
        if done:
            break
        actions.extend(new_actions)

    actions = actions[1:]
    while si < max_steps and si < width*depth and not done and len(actions) > 0:
        action = actions[0]
        actions = actions[1:]
        reward, done, new_actions = domain.step(action[0])
        si += 1
        metrics.append(domain.get_metrics())
        if done:
            break
        actions.extend(new_actions)
        si += 1

    metrics += [metrics[-1]] * (max_steps - len(metrics))

    return metrics
