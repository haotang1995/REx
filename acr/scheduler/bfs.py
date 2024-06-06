#!/usr/bin/env python
# coding=utf-8

def add_bfs_args(parser):
    parser.add_argument('--bfs-branching', type=int, default=3)
def get_bfs_args(args):
    return {
        'branching': args.bfs_branching,
    }

def bfs(
    domain, problem_id, seed,
    branching=3, max_steps=300,
):
    domain.set_seed(seed)

    actions = domain.reset(problem_id)
    assert len(actions) == 1, 'Grid domain must have exactly one init action'

    si = 0
    metrics = []
    while si < max_steps:
        action = actions[0]
        actions = actions[1:]
        for _ in range(branching):
            reward, done, new_actions = domain.step(action[0])
            metrics.append(domain.get_metrics())
            si += 1
            if done or si >= max_steps:
                break
            actions.extend(new_actions)
        if done:
            break
    assert si == len(metrics)

    metrics += [metrics[-1]] * (max_steps - len(metrics))
    return metrics
