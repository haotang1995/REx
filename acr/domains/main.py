#!/usr/bin/env python
# coding=utf-8

from .loop_inv import add_loop_inv_args, LoopInvDomain
from .arc import ARCDomain
from .apps import add_apps_args, APPSDomain

def add_domain_args(parser):
    add_loop_inv_args(parser)
    add_apps_args(parser)
    parser.add_argument('--domain', type=str, default='loop_inv',)

def get_domain(args):
    if args.domain == 'loop_inv':
        return LoopInvDomain(args)
    elif args.domain == 'arc':
        return ARCDomain(args)
    elif args.domain == 'apps':
        return APPSDomain(args)
    else:
        raise ValueError('Unknown domain: {}'.format(args.domain))

