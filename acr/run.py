#!/usr/bin/env python
# coding=utf-8

import os, sys
import argparse
import json

from .domains import add_domain_args, get_domain
from .scheduler import add_scheduler_args, run_with_scheduler
from .utils.llm import add_llm_args, get_llm
from .utils.logging import set_logger

def get_args():
    parser = argparse.ArgumentParser()
    add_domain_args(parser)
    add_scheduler_args(parser)
    add_llm_args(parser)
    parser.add_argument('--data_index', type=int, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    return args

def arg2name(args):
    args_str = ' '.join(sys.argv[1:])
    specified_args = {
        k: v
        for k, v in args.__dict__.items()
        if '-'+k in args_str or '-'+k.replace('_', '-') in args_str
    }
    name = '_'.join(''.join([p[0] for p in k.split('_')])+str(v) for k, v in specified_args.items())
    return name

def main():
    args = get_args()
    set_logger(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', arg2name(args)+'.log'))
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', args.domain)
    os.makedirs(result_dir, exist_ok=True)
    domain = get_domain(args)
    if args.data_index is not None:
        assert len(args.seeds) == 1, 'Cannot specify multiple seeds when running a single problem'
        metrics_list = run_with_scheduler(domain, args.data_index, args.seeds[0], args)
        print(metrics_list[-1])
        with open(os.path.join(result_dir, arg2name(args)+'.json'), 'w') as f:
            json.dump(metrics_list, f, indent=4)
    else:
        metrics = []
        for data_index in range(len(domain)):
            metrics.append([])
            for seed in args.seeds:
                metrics[-1].append(run_with_scheduler(domain, data_index, seed, args))
        # print(domain.summarize_results(metrics))
        with open(os.path.join(result_dir, arg2name(args)+'.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main()
