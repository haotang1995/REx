from .rex import add_rex_args, get_rex_args, rex
from .fw import add_fw_args, get_fw_args, fw
from .greedy import add_greedy_args, get_greedy_args, greedy
from .bfs import add_bfs_args, get_bfs_args, bfs

def add_scheduler_args(parser):
    add_rex_args(parser)
    add_fw_args(parser)
    add_greedy_args(parser)
    add_bfs_args(parser)
    parser.add_argument('--scheduler', type=str, default='rex')
    parser.add_argument('--max-steps', type=int, default=300)

def run_with_scheduler(domain, problem_id, seed, args):
    if args.scheduler == 'rex':
        return rex(domain, problem_id, seed, **get_rex_args(args), max_steps=args.max_steps)
    elif args.scheduler == 'fw':
        return fw(domain, problem_id, seed, **get_fw_args(args), max_steps=args.max_steps)
    elif args.scheduler == 'greedy':
        return greedy(domain, problem_id, seed, **get_greedy_args(args), max_steps=args.max_steps)
    elif args.scheduler == 'bfs':
        return bfs(domain, problem_id, seed, **get_bfs_args(args), max_steps=args.max_steps)
    else:
        raise ValueError(f'Unknown scheduler: {args.scheduler}')
