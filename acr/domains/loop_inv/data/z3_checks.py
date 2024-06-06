from z3 import *
from z3 import ToReal as _ToReal
import ast
import builtins
import copy
import time
import re
import subprocess
from subprocess import CalledProcessError

def get_solver(timeout=100):
    """
    Get a Z3 solver with a timeout.
    """
    miliseconds = timeout * 1000
    solver = Solver()
    solver.set('timeout', miliseconds)
    return solver

def timed_check(solver, timeout=100):
    # Force quit after time limit
    # timeout in seconds
    result, reason, model = None, None, None
    smt = solver.sexpr()

    file_dir = os.path.dirname(os.path.abspath(__file__))
    checker_file = file_dir + '/timed_check.py'
    try:
        checker_result = subprocess.check_output(['timeout', str(timeout), 'python3', checker_file, '-i', smt]).decode().split('\n', 1)
        result = checker_result[0]
        if result == "sat":
            model = checker_result[1][:-1]
    except CalledProcessError as e:
        result = "unknown"
        reason = "timeout"

    if result == "sat":
        result = sat
    elif result == "unsat":
        result = unsat
    elif result == "unknown":
        result = unknown

    parsed_result = {
        "result": result,
        "reason": reason,
        "model": model
    }

    return parsed_result
    # solver timing
    # start = time.time()
    # result = solver.check()
    # end = time.time()
    # print(f'Time taken to check the formula: {end-start} seconds')
    # if end-start > 2:
    #     print(solver)
    # return result

def find_vars(__expr):
    """
    Find all variables in the given expression.
    """
    assert isinstance(__expr, str), __expr
    try:
        __tree = ast.parse(__expr)
    except SyntaxError as se:
        # check if the error is due to real number parsing
        try:
            __tree = ast.parse(__expr.replace('k!', ''))
        except Exception as e:
            raise e
    except Exception as e:
        print(__expr)
        raise e
    __vars = set()
    for __node in ast.walk(__tree):
        if isinstance(__node, ast.Name):
            if __node.id not in globals() or (__node.id.isalpha() and len(__node.id)==1): # Not a z3 function
                if __node.id not in builtins.__dict__:
                    __vars.add(__node.id)
    if 'z3 ' in __expr:
        __vars.add('z3')
    return __vars

def init_exec_globals():
    """
    Initialize the globals() for the exec() function.
    """
    exec_globals = {k: v for k, v in globals().items() if not k.startswith('_')}
    return exec_globals

def init_vars(__vars, __exec_globals, real_vars=set()):
    """
    Initialize the z3 variables in the globals() for the exec() function.
    """
    for __var in __vars:
        if __var in real_vars:
            exec(f'{__var} = Real(\'{__var}\')', __exec_globals)
        else:
            exec(f'{__var} = Int(\'{__var}\')', __exec_globals)

def ToReal(x):
    if x.is_real():
        return x
    return _ToReal(x)
def eval_expr(__name, __expr, __exec_globals):
    """
    Evaluate the given expression with the given globals().
    """
    assert isinstance(__expr, str), f'__expr is not a string: {__expr}'
    assert isinstance(__exec_globals, dict), f'__exec_globals is not a dict: {__exec_globals}'
    exec(f'{__name} = {__expr}', __exec_globals)

def invtoinv2(inv, exec_globals, all_vars):
    try:
        _inv2 = str(inv).replace('\n', ' ')
        inv2 = ast.parse(_inv2)
    except SyntaxError as se:
        # check if the error is due to real number parsing
        try:
            inv2 = ast.parse(_inv2.replace('k!', ''))
        except Exception as e:
            raise e
    except Exception as e:
        print(_inv2)
        print(inv2)
        raise e
    for node in ast.walk(inv2):
        if isinstance(node, ast.Name):
            if node.id in all_vars:
                node.id += '2'
                assert node.id in all_vars, (node.id, all_vars)
    inv2 = ast.unparse(inv2)
    inv2 = re.sub(r'(\w+) != (\w+) ==', r'(\1 != \2) ==', inv2)
    inv2 = re.sub(r'== (\w+) >= (\w+)', r'== (\1 >= \2)', inv2)
    inv2 = inv2.replace('And()', 'True').replace('And(\n    )', 'True').replace('And(\n)', 'True')
    exec(f'inv2 = {inv2}', exec_globals)
    inv2 = exec_globals['inv2']
    return inv2

def _is_and_bool(z3_expr):
    try:
        return isinstance(z3_expr, BoolRef) and z3_expr.decl().kind() == Z3_OP_AND
    except Exception as e:
        print(z3_expr)
        print(e)
        return False
def check_per_loop_problem(loop_invariants, pre, cond, rec, post, real_vars=set(), timeout={}):
    """
    Check if the loop invariants are valid for the given pre, rec, and post conditions.
    For example, for the Cohen's U problem:
    * loop_invariants = ['z == 6 * n + 6', 'y == 3 * n * n + 3 * n + 1', 'x == n * n * n', 'n <= a + 1']
    * pre = 'And(n == 0, x == 0, y == 1, z == 6, a >= 0)'
    * cond = 'n <= a'
    * rec = 'And(n2 == n + 1, x2 == x + y, y2 == y + z, z2 == z + 6, a2 == a)'
    * post = 'And(x == (a + 1) * (a + 1) * (a + 1))'
    """
    assert isinstance(loop_invariants, (list, tuple)), type(loop_invariants)
    assert all(isinstance(inv, str) for inv in loop_invariants), loop_invariants
    assert isinstance(pre, str), pre
    assert isinstance(cond, str), cond
    assert isinstance(rec, str), rec
    assert isinstance(post, str), post

    # Preprocessing for corner cases, just to compensate for the bugs in Z3...
    # Replace '? != ? ==' with '(? != ?) =='
    loop_invariants = [re.sub(r'(\w+) != (\w+) ==', r'(\1 != \2) ==', inv) for inv in loop_invariants]
    loop_invariants = [re.sub(r'== (\w+) >= (\w+)', r'== (\1 >= \2)', inv) for inv in loop_invariants]
    loop_invariants = [inv.replace('And()', 'True').replace('And(\n    )', 'True').replace('And(\n)', 'True') for inv in loop_invariants]

    # Find all variables
    all_vars = set()
    all_vars.update(find_vars(pre))
    all_vars.update(find_vars(cond))
    all_vars.update(find_vars(post))
    assert not any(v.endswith('2') for v in all_vars), all_vars
    _all_vars = copy.deepcopy(all_vars)
    for inv in loop_invariants:
        inv_vars = find_vars(inv)
        inv_vars2 = {v+'2' for v in inv_vars}
        all_vars.update(inv_vars)
        all_vars.update(inv_vars2)
    # assert not any(v+'2' in all_vars for v in _all_vars), (all_vars, _all_vars)
    all_vars.update(find_vars(rec))
    all_vars.discard('z3')

    # Create all z3 variables in locals()
    exec_globals = init_exec_globals()
    init_vars(all_vars, exec_globals, real_vars)
    vars_in_z3 = {v: exec_globals[v] for v in all_vars}
    for vi, inv in enumerate(loop_invariants):
        eval_expr(f'__inv{vi}', inv, exec_globals)
        exec_globals.update(vars_in_z3)
    eval_expr('pre_z3', pre, exec_globals)
    eval_expr('cond_z3', cond, exec_globals)
    eval_expr('rec_z3', rec, exec_globals)
    eval_expr('post_z3', post, exec_globals)
    for var in all_vars:
        locals()[var] = exec_globals[var]
    invariants_z3 = [exec_globals[f'__inv{i}'] for i in range(len(loop_invariants))]
    pre_z3, cond_z3, rec_z3, post_z3 = exec_globals['pre_z3'], exec_globals['cond_z3'], exec_globals['rec_z3'], exec_globals['post_z3']

    # Simplify the invariants and split them into separate conditions
    _invariants_z3 = simplify(And(*invariants_z3))
    solver = get_solver()
    solver.add(_invariants_z3)
    result = timed_check(solver)['result']
    if result == sat:
        invariants_z3 = _invariants_z3
        if _is_and_bool(invariants_z3):
            invariants_z3 = list(invariants_z3.children())
        else:
            invariants_z3 = [invariants_z3]
    # else:
        # print('Failed to simplify the invariants')
        # print(_invariants_z3)

    # Check if the invariants are valid
    per_inv_metrics = [dict() for _ in invariants_z3]
    for i, inv in enumerate(invariants_z3):
        # Check pre -> inv
        solver = get_solver() if "pre" not in timeout else get_solver(timeout["pre"])
        # print(f'Checking inv{i} establish, {Not(Implies(pre_z3, inv))}')
        solver.add(Not(Implies(pre_z3, inv)))
        #print("Pre -> Inv")
        #print(solver.sexpr())
        result_dict = timed_check(solver) if "pre" not in timeout else timed_check(solver, timeout["pre"])
        result = result_dict['result']
        per_inv_metrics[i]['establish'] = {
            'success': result == unsat,
            '_checker_result': result,
            '_checker_model': result_dict['model'] if result == sat else None,
            'assertion': str(Implies(pre_z3, inv)),
        }

        # Check loop_invs, cond, rec -> inv2
        solver = get_solver() if "trans" not in timeout else get_solver(timeout["trans"])
        inv2 = invtoinv2(inv, exec_globals, all_vars)
        # print(f'Checking inv{i} preserve, {Not(Implies(And(*invariants_z3, cond_z3, rec_z3), inv2))}')
        solver.add(Not(Implies(And(*invariants_z3, cond_z3, rec_z3), inv2)))
        #print("Loop Invs, Cond, Rec -> Inv2")
        #print(solver.sexpr())
        result_dict = timed_check(solver) if "trans" not in timeout else timed_check(solver, timeout["trans"])
        result = result_dict['result']
        per_inv_metrics[i]['preserve'] = {
            'success': result == unsat,
            '_checker_result': result,
            '_checker_model': result_dict['model'] if result == sat else None,
            'assertion': str(Implies(And(*invariants_z3, cond_z3, rec_z3), inv2)),
        }

        per_inv_metrics[i]['inductive'] = per_inv_metrics[i]['establish']['success'] and per_inv_metrics[i]['preserve']['success']

    metrics = {'per_inv': per_inv_metrics}
    metrics['post'] = dict()
    # Check loop_invs, not cond -> post
    solver = get_solver() if "post" not in timeout else get_solver(timeout["post"])
    # print(f'Checking post, {Not(Implies(And(*invariants_z3, Not(cond_z3)), post_z3))}')
    solver.add(Not(Implies(And(*invariants_z3, Not(cond_z3)), post_z3)))
    #print("Loop Invs, Not Cond -> Post")
    #print(solver.sexpr())
    result_dict = timed_check(solver) if "post" not in timeout else timed_check(solver, timeout["post"])
    result = result_dict['result']
    metrics['post']['success'] = result == unsat
    metrics['post']['overall'] = {
        'success': result == unsat,
        '_checker_result': result,
        '_checker_model': result_dict['model'] if result == sat else None,
        'assertion': str(Implies(And(*invariants_z3, Not(cond_z3)), post_z3)),
    }

    simplied_post_z3 = simplify(post_z3)
    if _is_and_bool(simplied_post_z3):
        post_children = simplied_post_z3.children()
    else:
        post_children = [simplied_post_z3]
    post_children_result = []
    for post_child in post_children:
        solver = get_solver() if "post" not in timeout else get_solver(timeout["post"])
        # print(f'Checking post child, {Not(Implies(And(*invariants_z3, Not(cond_z3)), post_child))}')
        solver.add(Not(Implies(And(*invariants_z3, Not(cond_z3)), post_child)))
        result_dict = timed_check(solver) if "post" not in timeout else timed_check(solver, timeout["post"])
        result = result_dict['result']
        post_children_result.append({
            'success': result == unsat,
            '_checker_result': result,
            '_checker_model': result_dict['model'] if result == sat else None,
            'assertion': str(Implies(And(*invariants_z3, Not(cond_z3)), post_child)),
        })
    metrics['post']['per_child'] = post_children_result
    metrics['post']['success_rate'] = sum(m['success'] for m in post_children_result) / len(post_children_result)

    metrics['establish'] = {
        'success': all(m['establish']['success'] for m in per_inv_metrics),
        'success_rate': sum(m['establish']['success'] for m in per_inv_metrics) / len(per_inv_metrics),
    }
    metrics['preserve'] = {
        'success': all(m['preserve']['success'] for m in per_inv_metrics),
        'success_rate': sum(m['preserve']['success'] for m in per_inv_metrics) / len(per_inv_metrics),
    }
    metrics['inductive'] = {
        'success': all(m['inductive'] for m in per_inv_metrics),
        'success_rate': sum(m['inductive'] for m in per_inv_metrics) / len(per_inv_metrics),
    }
    metrics['success'] = metrics['inductive']['success'] and metrics['post']['success']

    simplified_loop_invariants = [str(inv).replace('\n', ' ') for inv in invariants_z3]
    simplified_loop_invariants = [re.sub(r'(\w+) != (\w+) ==', r'(\1 != \2) ==', inv) for inv in simplified_loop_invariants]
    simplified_loop_invariants = [re.sub(r'== (\w+) >= (\w+)', r'== (\1 >= \2)', inv) for inv in simplified_loop_invariants]
    simplified_loop_invariants = [inv.replace('And()', 'True').replace('And(\n    )', 'True').replace('And(\n)', 'True') for inv in simplified_loop_invariants]

    output = {
        'success': metrics['success'],
        'overall_metrics': {
            'establish': metrics['establish'],
            'preserve': metrics['preserve'],
            'inductive': metrics['inductive'],
            'post': metrics['post'],
        },
        'per_inv_metrics': per_inv_metrics,
        'inputs': {
            'pre': pre,
            'cond': cond,
            'rec': rec,
            'post': post,
            'loop_invariants': loop_invariants,
        },
        'intermediate': {
            'simplified_loop_invariants': simplified_loop_invariants,
        },
    }
    return output, simplified_loop_invariants

# T1 = 2
# T2 = 16
# T3 = 100
T1, T2, T3 = 100, 100, 100

PROBLEMS_IN_Z3 = {
    # 0
    ('cohencu', 1): {
        'vars': ['x', 'y', 'z', 'n', 'a'],
        'cond': 'n <= a',
        'pre': 'And(n == 0, x == 0, y == 1, z == 6, a >= 0)',
        'rec': 'And(n2 == n + 1, x2 == x + y, y2 == y + z, z2 == z + 6, a2 == a)',
        'post': 'And(x == (a + 1) * (a + 1) * (a + 1))',
        'timeout': {'pre': T1, 'trans': T2, 'post': T1},
    },
    # 1
    ('cohendiv', 1): {
        'vars': ['x', 'y', 'q', 'r', 'a', 'b'],
        'cond': 'r >= y',
        'pre': 'And(x > 0, y > 0, q == 0, r == x, a == 0, b == 0)',
        # using loop2 post -- inv: r2 >= b2, b2 == y2 * a2, a2 >= 1,  lc: r2 >= 2 * b2;  then (r=r-b, q=q+a)
        'rec': 'And(x2 == x, y2 == y, b2 == y2 * a2, a2 >= 1, r2 == r - b2, q2 == q + a2, r2 >= 0, r2 < b2)',
        'post': 'And(x == q * y + r, r >= 0, r < y)',
        # Need to show recall on (x>=1, y>=1, x == q * y + r, r >= 0), the supposed loop inv
        'req_inv': 'And(x>=1, y>=1, x == q * y + r, r >= 0)',
        'timeout': {'pre': T1, 'trans': T2, 'post': T1},
    },
    # 2
    ('cohendiv', 2): {
        'vars': ['x', 'y', 'q', 'r', 'a', 'b'],
        'cond': 'r >= 2*b',
        # loop1 inv: x>=1, y>=1, x == q * y + r, r >= 0
        'pre': 'And(a == 1, b == y, r >= y, r >= 0, x >= 1, y >= 1, x == q * y + r)',   # x == q * y + r, r >= 0, x >= 1, y >= 1,
        'rec': 'And(x2 == x, y2 == y, q2 == q, r2 == r, a2 == 2 * a, b2 == 2 * b)',
        'post': 'And(r >= b, b == y * a, a >= 1, r < 2 * b)',   # matches loop1 rec   , Or(a == 1, a % 2 == 0)
        # 'post': 'And(r >= b, b == y * a, x == q * y + r, r >= 0, x >= 1, y >= 1, r < 2 * b)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 3
    ('dijkstra', 1): {
        'vars': ['p', 'q', 'r', 'h', 'n'],
        'cond': 'q <= n',
        'pre': 'And(p == 0, q == 1, r == n, h == 0, n >= 0)',
        'rec': 'And(p2 == p, q2 == 4 * q, r2 == r, n2 == n, h2 == h)',
        'post': 'And(p == 0, q > n, r == n, h == 0, n >= 0)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 4
    ('dijkstra', 2): {
        'vars': ['p', 'q', 'r', 'h', 'n'],
        'cond': 'q != 1',
        'pre': 'And(p == 0, q > n, r == n, h == 0, n >= 0)',
        'rec': 'And(q2 == q / 4, q == 4 * q2, h2 == p + q2, n2 == n, Or(And(r >= h2, p2 == p / 2 + q2, p == 2 * p2 - 2 * q2, r2 == r - h2), And(r < h2, p2 == p / 2, p == 2 * p2, r2 == r)))',
        'post': 'And(p * p <= n, (p + 1) * (p + 1) > n)',
        'timeout': {'pre': T1, 'trans': T2, 'post': T1},
    },
    # 5
    ('divbin', 1): {
        'vars': ['A', 'B', 'q', 'r', 'b'],
        'cond': 'r >= b',
        'pre': 'And(A > 0, B > 0, q == 0, r == A, b == B)',
        'rec': 'And(A2 == A, B2 == B, q2 == q, r2 == r, b2 == 2 * b)',
        'post': 'And(q == 0, A == r, b > 0, r > 0, r < b)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 6
    ('divbin', 2): {
        'vars': ['A', 'B', 'q', 'r', 'b'],
        'cond': 'b != B',
        'pre': 'And(q == 0, A == r, b > 0, r > 0, r < b)',
        'rec': 'And(A2 == A, B2 == B, b2 == b / 2, b == 2 * b2, Or(And(r >= b2, q2 == 2 * q + 1, r2 == r - b2), And(r < b2, q2 == 2 * q, r2 == r)))',
        'post': 'And(A == q * B + r, r >= 0, r < B,)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 7
    ('egcd', 1): {
        'vars': ['x', 'y', 'a', 'b', 'p', 'q', 'r', 's'],
        'cond': 'a != b',
        'pre': 'And(x >= 1, y >= 1, a == x, b == y, p == 1, q == 0, r == 0, s == 1)',
        'rec': 'And(x2 == x, y2 == y, Or(And(a > b, a2 == a - b, p2 == p - q, r2 == r - s, b2 == b, q2 == q, s2 == s), And(a <= b, b2 == b - a, q2 == q - p, s2 == s - r, a2 == a, p2 == p, r2 == r)))',
        'post': 'And(1 == p * s - r * q, a == y * r + x * p, b == x * q + y * s, a == b,  a > 0, b > 0, x > 0, y > 0)',
        'timeout': {'pre': T1, 'trans': T2, 'post': T2},
    },
    # 8
    ('egcd2', 1): {
        'vars': ['x', 'y', 'a', 'b', 'p', 'q', 'r', 's', 'k', 'c'],
        'cond': 'b != 0',
        'pre': 'And(x >= 1, y >= 1, a == x, b == y, p == 1, q == 0, r == 0, s == 1)',
        # using loop2 post: a == quot * b + rem, rem >= 0, rem < b, quot >= 0, rem == a % b, quot == a // b
        'rec': 'And(x2 == x, y2 == y, a == k * b + c, c >= 0, c < b, k >= 0, p2 == q, q2 == p - k * q, r2 == s, s2 == r - k * s, a2 == b, b2 == c)', # rem == a % b, quot == a // b
        'post': 'And(Or(1 == p * s - r * q, 1 == r * q - p * s), a == y * r + x * p, b == x * q + y * s, b == 0, Or(a * s == x, a * s == - x), Or(a * q == y, a * q == - y))', # Extended Euclid Alg
        # need to show recall on
        'req_inv': 'And(a >= 0, b >= 0)',
        'timeout': {'pre': T1, 'trans': T3, 'post': T3},
    },
    # 9
    ('egcd2', 2): {
        'vars': ['x', 'y', 'a', 'b', 'p', 'q', 'r', 's', 'k', 'c'],
        'cond': 'c >= b',
        # using loop1 inv:  a >= 0, b >= 0
        'pre': 'And(b != 0, k == 0, b >= 0, a >= 0, c == a)',
        'rec': 'And(x2 == x, y2 == y, p2 == p, r2 == r, q2 == q, s2 == s, a2 == a, b2 == b, c2 == c - b, k2 == k + 1)',
        'post': 'And(a == k * b + c, c >= 0, c < b, k >= 0)',   # matches loop1 rec
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 10
    ('egcd3', 1): {
        'vars': ['x', 'y', 'a', 'b', 'p', 'q', 'r', 's', 'k', 'c', 'd', 'v'],
        'cond': 'b != 0',
        'pre': 'And(x >= 1, y >= 1, a == x, b == y, p == 1, q == 0, r == 0, s == 1)',
        # using loop2 post: a == quot * b + rem, rem >= 0, rem < b, quot >= 0, rem == a % b, quot == a // b
        'rec': 'And(x2 == x, y2 == y, a == k * b + c, c >= 0, c < b, k >= 0, p2 == q, q2 == p - k * q, r2 == s, s2 == r - k * s, a2 == b, b2 == c)', # rem == a % b, quot == a // b
        'post': 'And(Or(1 == p * s - r * q, 1 == r * q - p * s), a == y * r + x * p, b == x * q + y * s, b == 0, Or(a * s == x, a * s == - x), Or(a * q == y, a * q == - y))', # Extended Euclid Alg
        # need to show recall on
        'req_inv': 'And(a >= 0, b >= 0)',
        'timeout': {'pre': T1, 'trans': T3, 'post': T3},
    },
    # 11
    ('egcd3', 2): {
        'vars': ['x', 'y', 'a', 'b', 'p', 'q', 'r', 's', 'k', 'c', 'd', 'v'],
        'cond': 'c >= b',
        # using loop1 inv:  a >= 0, b >= 0
        'pre': 'And(b != 0, k == 0, b >= 0, a >= 0, c == a)',
        # using loop3 post: c >= v, v == b * d, d >= 1, c < 2 * v
        'rec': 'And(x2 == x, y2 == y, p2 == p, r2 == r, q2 == q, s2 == s, a2 == a, b2 == b, c2 == c - v2, k2 == k + d2, c2 >= 0, v2 == b2 * d2, d2 >= 1, c2 < v2)',
        'post': 'And(a == k * b + c, c >= 0, c < b, k >= 0)',   # matches loop1 rec
        # need to show recall on
        'req_inv': 'And(c >= 0, b >= 0)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 12
    ('egcd3', 3): {
        'vars': ['x', 'y', 'a', 'b', 'p', 'q', 'r', 's', 'k', 'c', 'd', 'v'],
        'cond': 'c >= 2 * v',
        # using loop2 inv:  c >= 0, b >= 0
        'pre': 'And(b != 0, c >= 0, b >= 0, c >= b, d == 1, v == b)',
        'rec': 'And(x2 == x, y2 == y, p2 == p, r2 == r, q2 == q, s2 == s, a2 == a, b2 == b, c2 == c, k2 == k, d2 == 2 * d, v2 == 2 * v)',
        'post': 'And(c >= v, v == b * d, d >= 1, c < 2 * v)',   # matches loop2 rec
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 13
    ('fermat1', 1): {
        'vars': ['A', 'R', 'u', 'v', 'r'],
        'cond': 'r != 0',
        'pre': 'And(A >= 1, (R - 1) * (R - 1) < A, A <= R * R, A % 2 == 1, u == 2 * R + 1, v == 1, r == R * R - A)',
        'rec': 'And(A2 == A, R2 == R,  4 * (r2 + A2) == (u2*u2) - (v2*v2) - 2*u2 + 2*v2, v2 % 2 == 1, u2 % 2 == 1)',
        'post': 'A == ((u-v)/2) * ((u+v-2)/2)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T3},
    },
    # 14
    ('fermat1', 2): {
        'vars': ['A', 'R', 'u', 'v', 'r'],
        'cond': 'r > 0',
        'pre': 'And(r != 0, v % 2 == 1, v > 0)', # loop1 inv
        'rec': 'And(r2 == r - v, v2 == v + 2, A2 == A, R2 == R, u2 == u)',
        'post': 'And(v % 2 == 1)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 15
    ('fermat1', 3): {
        'vars': ['A', 'R', 'u', 'v', 'r'],
        'cond': 'r < 0',
        'pre': 'And(u % 2 == 1, u > 0)', # loop2 post
        'rec': 'And(r2 == r + u, u2 == u + 2, A2 == A, R2 == R, v2 == v)',
        'post': 'And(u % 2 == 1)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 16
    ('fermat2', 1): {
        'vars': ['A', 'R', 'u', 'v', 'r'],
        'cond': 'r != 0',
        'pre': 'And(A >= 1, (R - 1) * (R - 1) < A, A <= R * R, A % 2 == 1, u == 2 * R + 1, v == 1, r == R * R - A)',
        'rec': 'And(A2 == A, R2 == R, Or(And(r > 0, u2 == u, v2 == v + 2, r2 == r - v), And(r <= 0, u2 == u + 2, v2 == v, r2 == r + u)))',
        'post': 'A == ((u-v)/2) * ((u+v-2)/2)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T3},
    },
    # 17
    ('freire1', 1): {
        'vars': ['a', 'x', 'r'],
        'cond': 'x > r',
        'pre': 'And(a > 0, x == ToReal(a) / 2.0, r == 0)',
        'rec': 'And(x2 == x - r, r2 == r + 1, a2 == a)',
        'post': 'And(r * r + r >= a, r * r - r < a)',
        'real_vars': {'x'},
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 18
    ('freire2', 1): {
        'vars': ['a', 'x', 'r', 's'],
        # Hard problem for z3
        'cond': 'x - s > 0',
        'pre': 'And(a > 0, x == ToReal(a), s == 3.25, r == 1)',
        'rec': 'And(a2 == a, x2 == x - s, s2 == s + 6 * r + 3, r2 == r + 1)',
        'post': 'And(4 * r * r * r + 6 * r * r + 3 * r >= 4 * a, 4 * a > 4 * r * r * r - 6 * r * r + 3 * r - 1)',
        'real_vars': {'x', 's'},
        'timeout': {'pre': T1, 'trans': T3, 'post': T3},
    },
    ('freire2', 1): {
        'vars': ['a', 'x', 'r', 's'],
        # If Real is too slow, use this
        'cond': 'x - s > 0',
        'pre': 'And(a > 0, x == a, s == 3.25, r == 1)',
        'rec': 'And(a2 == a, x2 == x - s, s2 == s + 6 * r + 3, r2 == r + 1)',
        'post': 'And(4 * r * r * r + 6 * r * r + 3 * r >= 4 * a, 4 * a > 4 * r * r * r - 6 * r * r + 3 * r - 1, \
                4*r*r*r - 6*r*r + 3*r + 4*x - 4*a == 1, 4*s - 12*r*r == 1, x > 0)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 19
    ('geo1', 1): {
        'vars': ['z', 'k', 'c', 'x', 'y'],
        'cond': 'c < k',
        'pre': 'And(z >= 0, z <= 10, k > 0, k <= 10, c == 1, x == 1, y == z)',
        'rec': 'And(z2 == z, k2 == k, c2 == c + 1, x2 == x * z + 1, y2 == y * z)',
        'post': 'x * (z - 1) == y - 1',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 20
    ('geo2', 1): {
        'vars': ['z', 'k', 'c', 'x', 'y'],
        'cond': 'c < k',
        'pre': 'And(z >= 0, z <= 10, k > 0, k <= 10, c == 1, x == 1, y == 1)',
        'rec': 'And(z2 == z, k2 == k, c2 == c + 1, x2 == x * z + 1, y2 == y * z)',
        'post': 'x * (z - 1) == z * y - 1',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 21
    ('geo3', 1): {
        'vars': ['z', 'k', 'c', 'x', 'y', 'a'],
        'cond': 'c < k',
        'pre': 'And(z >= 0, z <= 10, k > 0, k <= 10, c == 1, x == a, y == 1)',
        'rec': 'And(z2 == z, k2 == k, a2 == a, c2 == c + 1, x2 == x * z + a, y2 == y * z)',
        'post': 'x * z - x + a - a * z * y == 0',
        'timeout': {'pre': T1, 'trans': T2, 'post': T1},
    },
    # 22
    ('hard', 1): {
        'vars': ['A', 'B', 'r', 'd', 'p', 'q'],
        'cond': 'r >= d',
        'pre': 'And(A >= 0, B > 0, r == A, d == B, p == 1, q == 0)',
        'rec': 'And(A2 == A, B2 == B, d2 == 2 * d, p2 == 2 * p, q2 == q, r2 == r)',
        'post': 'And(A >= 0, B > 0, r == A, d == B * p, q == 0, r < d)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 23
    ('hard', 2): {
        'vars': ['A', 'B', 'r', 'd', 'p', 'q'],
        'cond': 'p != 1',
        'pre': 'And(A >= 0, B > 0, r == A, d == B * p, q == 0, r < d)',
        'rec': 'And(A2 == A, B2 == B, d == d2 * 2, p == p2 * 2, Or(And(r >= d2, r2 == r - d2, q2 == q + p2), And(r < d2, q2 == q, r2 == r)))',
        'post': 'And(A == q * B + r, r >= 0, r < B)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 24
    ('knuth', 1): {
        'vars': ['n', 'a', 'd', 'r', 'k', 'q', 's', 't'],
        'cond': 'And(s >= d, r != 0)',
        'pre': 'And(n > 0, n % 2 == 1, a % 2 == 1, (a-1)*(a-1)*(a-1) < 8 * n, d == a, r == n % d, k == n % (d - 2), q == 4 * (n/(d-2) - n/d), s * s <= n, (s+1)*(s+1) > n)',
        'rec': 'And( n2 == n, a2 == a, s2 == s,  Or( \
                    And(2*r-k+q<0,                                  t2 == r, r2 == 2*r-k+q+d+2,  k2 == r, q2 == q + 4, d2 == d + 2), \
                    And(2*r-k+q>=0, 2*r-k+q<d+2,                    t2 == r, r2 == 2*r-k+q,      k2 == r, q2 == q, d2 == d + 2), \
                    And(2*r-k+q>=0, 2*r-k+q>=d+2, 2*r-k+q<2*d+4,    t2 == r, r2 == 2*r-k+q-d-2,  k2 == r, q2 == q - 4, d2 == d + 2), \
                    And(2*r-k+q>=0, 2*r-k+q>=2*d+4,                 t2 == r, r2 == 2*r-k+q-2*d-4, k2 == r, q2 == q - 8, d2 == d + 2) \
                ))',
        'post': 'Implies(r == 0, d * (d * q - 2 * q + 4 * k) == 8 * n)', # n % d == 0 )',
        'timeout': {'pre': T2, 'trans': T2, 'post': T1},
    },
    # 25
    ('lcm1', 1): {
        'vars': ['x', 'y', 'u', 'v', 'a', 'b', 'p', 'q', 'r', 's'],
        'cond': 'x != y',
        'pre': 'And(a > 0, b > 0, x == a, y == b, u == b, v == 0,  p == 1, q == 0, r == 0, s == 1)',
        'rec': 'And(a2 == a, b2 == b, Or( \
                    And(x > y, x2 == x - y, y2 == y, u2 == u, v2 == v + u,  p2 == p - q, r2 == r - s, q2 == q, s2 == s), \
                    And(x <= y, x2 == x, y2 == y - x, u2 == u + v, v2 == v, q2 == q - p, s2 == s - r, p2 == p, r2 == r) \
                ))',
        # a * b == GCD * LCM, x == GCD
        'post': 'And((u + v) * x == a * b, 1 == p * s - r * q, x == b * r + a * p, y == a * q + b * s, x == y )',
        # Exists([p, q, r, s], And(1 == p * s - r * q, x == r * b + p * a, y == q * a + s * b))
        'timeout': {'pre': T1, 'trans': T3, 'post': T2},
    },
    # 26
    ('lcm1', 2): {
        # only verify LCM part
        'vars': ['x', 'y', 'u', 'v', 'a', 'b', 'p', 'q', 'r', 's'],
        'cond': 'x > y',
        'pre': 'x*u + y*v == a*b',
        'rec': 'And(a2 == a, b2 == b, x2 == x - y, y2 == y, u2 == u, v2 == v + u, p2 == p - q, q2 == q, r2 == r - s, s2 == s)',
        'post': 'x*u + y*v == a*b',
        'timeout': {'pre': T1, 'trans': T2, 'post': T1},
    },
    # 27
    ('lcm1', 3): {
        # only verify LCM part
        'vars': ['x', 'y', 'u', 'v', 'a', 'b', 'p', 'q', 'r', 's'],
        'cond': 'x < y',
        'pre': 'x*u + y*v == a*b',
        'rec': 'And(a2 == a, b2 == b, x2 == x, y2 == y - x, u2 == u + v, v2 == v, p2 == p, q2 == q - p, r2 == r, s2 == s - r)',
        'post': 'x*u + y*v == a*b',
        'timeout': {'pre': T1, 'trans': T2, 'post': T1},
    },
    # 28
    ('lcm2', 1): {
        'vars': ['x', 'y', 'u', 'v', 'a', 'b', 'p', 'q', 'r', 's'],
        'cond': 'x != y',
        'pre': 'And(a > 0, b > 0, x == a, y == b, u == b, v == a,  p == 1, q == 0, r == 0, s == 1)',
        'rec': 'And(a2 == a, b2 == b, Or( \
                    And(x > y, x2 == x - y, y2 == y, u2 == u, v2 == v + u,  p2 == p - q, r2 == r - s, q2 == q, s2 == s), \
                    And(x <= y, x2 == x, y2 == y - x, u2 == u + v, v2 == v, q2 == q - p, s2 == s - r, p2 == p, r2 == r) \
                ))',
        # a * b == GCD * LCM, x == GCD
        'post': 'And((u + v) * x == 2 * a * b, 1 == p * s - r * q, x == b * r + a * p, y == a * q + b * s, x == y )',
        # Exists([p, q, r, s], And(1 == p * s - r * q, x == r * b + p * a, y == q * a + s * b))
        'timeout': {'pre': T1, 'trans': T3, 'post': T2},
    },
    # 29
    ('mannadiv', 1): {
        'vars': ['A', 'B', 'q', 'r', 't'],
        'cond': 't != 0',
        'pre': 'And(A >= 0, B > 0, q ==0, r == 0, t == A)',
        'rec': 'And(A2 == A, B2 == B, Or(And(r + 1 == B, q2 == q + 1, r2 == 0, t2 == t - 1), And(r + 1 != B, q2 == q, r2 == r + 1, t2 == t - 1)))',
        'post': 'And(A == q * B + r, r >= 0, r < B)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 30
    ('prod4br', 1): {
        'vars': ['x', 'y', 'a', 'b', 'p', 'q'],
        'cond': 'And(a != 0, b != 0)',
        'pre': 'And(x >= 0, y >= 0, a == x, b == y, p == 1, q == 0)',
        'rec': 'And(x2 == x, y2 == y, \
                Or( And(a % 2 == 0, b % 2 == 0, a2 == a/2, b2 == b/2, p2 == 4 * p, q2 == q), \
                    And(a % 2 == 1, b % 2 == 0, a2 == a - 1, b2 == b, p2 == p, q2 == q + b * p), \
                    And(a % 2 == 0, b % 2 == 1, a2 == a, b2 == b - 1, p2 == p, q2 == q + a * p), \
                    And(a % 2 == 1, b % 2 == 1, a2 == a - 1, b2 == b - 1, p2 == p, q2 == q + (a2 + b2 + 1) * p) \
                ))',
        'post': 'q == x * y',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 31
    ('prodbin', 1): {
        'vars': ['a', 'b', 'x', 'y', 'z'],
        'cond': 'y != 0',
        'pre': 'And(a >= 0, b >= 0, x == a, y == b, z == 0)',
        'rec': 'And(a2 == a, b2 == b, Or( \
                    And(y % 2 == 1, z2 == z + x, x2 == x * 2, y2 == (y-1) / 2), \
                    And(y % 2 == 0, z2 == z, x2 == x * 2, y2 == y / 2) \
                ))',
        'post': 'z == a * b',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 32
    ('ps2', 1): {
        'vars': ['k', 'y', 'x', 'c'],
        'cond': 'c < k',
        'pre': 'And(k >= 0, k <= 30, y == 0, x == 0, c == 0)',
        'rec': 'And(k2 == k, c2 == c + 1, y2 == y + 1, x2 == x + y2)',
        'post': 'x == k * (k + 1) / 2',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 33
    ('ps3', 1): {
        'vars': ['k', 'y', 'x', 'c'],
        'cond': 'c < k',
        'pre': 'And(k >= 0, k <= 30, y == 0, x == 0, c == 0)',
        'rec': 'And(k2 == k, c2 == c + 1, y2 == y + 1, x2 == x + y2 * y2)',
        'post': 'x == k * (k + 1) * (2 * k + 1) / 6',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
    # 34
    ('ps4', 1): {
        'vars': ['k', 'y', 'x', 'c'],
        'cond': 'c < k',
        'pre': 'And(k >= 0, k <= 30, y == 0, x == 0, c == 0)',
        'rec': 'And(k2 == k, c2 == c + 1, y2 == y + 1, x2 == x + y2 * y2 * y2)',
        'post': 'x == k * k * (k + 1) * (k + 1) / 4',
        'timeout': {'pre': T1, 'trans': T1, 'post': T2},
    },
    # 35
    ('ps5', 1): {
        'vars': ['k', 'y', 'x', 'c'],
        'cond': 'c < k',
        'pre': 'And(k >= 0, k <= 30, y == 0, x == 0, c == 0)',
        'rec': 'And(k2 == k, c2 == c + 1, y2 == y + 1, x2 == x + y2 * y2 * y2 * y2)',
        'post': '30 * x == 6 * k * k * k * k * k + 15 * k * k * k * k + 10 * k * k * k - k',
        'timeout': {'pre': T1, 'trans': T2, 'post': T2},
    },
    # 36
    ('ps6', 1): {
        'vars': ['k', 'y', 'x', 'c'],
        'cond': 'c < k',
        'pre': 'And(k >= 0, k <= 30, y == 0, x == 0, c == 0)',
        'rec': 'And(k2 == k, c2 == c + 1, y2 == y + 1, x2 == x + y2 * y2 * y2 * y2 * y2)',
        'post': '12 * x == 2 * k * k * k * k * k * k + 6 * k * k * k * k * k + 5 * k * k * k * k - k * k',
        'timeout': {'pre': T1, 'trans': T2, 'post': T2},
    },
    # 37
    ('sqrt1', 1): {
        'vars': ['n', 'a', 's', 't', 'ctr'],
        'cond': 's <= n',
        'pre': 'And(n >= 0, a == 0, s == 1, t == 1, ctr == 0)',
        'rec': 'And(n2 == n, a2 == a + 1, t2 == t + 2, s2 == s + t2)',
        'post': 'And(a * a <= n, (a + 1) * (a + 1) > n)',
        'timeout': {'pre': T1, 'trans': T1, 'post': T1},
    },
}

def check_z3_problem(problem_name, loop_index, loop_invariants,):
    assert (problem_name, loop_index) in PROBLEMS_IN_Z3, (problem_name, loop_index)
    problem = PROBLEMS_IN_Z3[(problem_name, loop_index)]
    return check_per_loop_problem(
        loop_invariants=loop_invariants,
        pre=problem['pre'],
        cond=problem['cond'],
        rec=problem['rec'],
        post=problem['post'],
        real_vars=problem.get('real_vars', set()),
        timeout=problem['timeout'],
    )

def compare_with_ground_truth(loop_invs, true_loop_invs):
    all_vars = set()
    for inv in loop_invs:
        all_vars.update(find_vars(inv))
    for inv in true_loop_invs:
        all_vars.update(find_vars(inv))
    exec_globals = init_exec_globals()
    init_vars(all_vars, exec_globals)
    for vi, inv in enumerate(loop_invs):
        eval_expr(f'__inv{vi}', inv, exec_globals)
    loop_invs_z3 = [exec_globals[f'__inv{i}'] for i in range(len(loop_invs))]
    for vi, inv in enumerate(true_loop_invs):
        eval_expr(f'__inv{vi}', inv, exec_globals)
    true_loop_invs_z3 = [exec_globals[f'__inv{i}'] for i in range(len(true_loop_invs))]

    # Simplify the invariants and split them into separate conditions
    loop_invs_z3 = simplify(And(*loop_invs_z3))
    if _is_and_bool(loop_invs_z3):
        loop_invs_z3 = list(loop_invs_z3.children())
    else:
        loop_invs_z3 = [loop_invs_z3]
    simplied_loop_invs = [str(inv).replace('\n', ' ') for inv in loop_invs_z3]
    assert len(set(simplied_loop_invs)) == len(simplied_loop_invs), simplied_loop_invs
    true_loop_invs_z3 = simplify(And(*true_loop_invs_z3))
    if _is_and_bool(true_loop_invs_z3):
        true_loop_invs_z3 = list(true_loop_invs_z3.children())
    else:
        true_loop_invs_z3 = [true_loop_invs_z3]
    simplied_true_loop_invs = [str(inv).replace('\n', ' ') for inv in true_loop_invs_z3]
    assert len(set(simplied_true_loop_invs)) == len(simplied_true_loop_invs), simplied_true_loop_invs

    # Old Naive way to calculate precision and recall @deprecated on 2024-04-10
    # # Remove equivalent invariants
    # unique_invariants = [True] * len(loop_invs_z3)
    # for i, inv in enumerate(loop_invs_z3):
        # if not unique_invariants[i]:
            # continue
        # for j in range(i+1, len(loop_invs_z3)):
            # inv2 = loop_invs_z3[j]
            # solver = get_solver()
            # solver.add(Not(And(Implies(inv, inv2), Implies(inv2, inv))))
            # result = timed_check(solver)['result']
            # if result == unsat:
                # unique_invariants[j] = False
    # loop_invs_z3 = [inv for i, inv in enumerate(loop_invs_z3) if unique_invariants[i]]

    # Check if the invariants are equal
    # correctness = []
    # for inv in loop_invs_z3:
        # correctness.append(False)
        # if inv in true_loop_invs_z3:
            # correctness[-1] = True
        # else:
            # for true_inv in true_loop_invs_z3:
                # solver = get_solver()
                # solver.add(Not(And(Implies(inv, true_inv), Implies(true_inv, inv))))
                # result = timed_check(solver)['result']
                # if result == unsat:
                    # correctness[-1] = True
                    # break

    # precision = sum(correctness) / len(correctness)
    # recall = sum(correctness) / len(true_loop_invs_z3)


    # New way to calculate precision and recall
    unified_invs = And(*loop_invs_z3)
    unified_true_invs = And(*true_loop_invs_z3)

    found_true_invs = []
    solver = get_solver()
    solver.add(unified_invs)
    result = timed_check(solver)['result']
    if result == sat:
        for tinv in true_loop_invs_z3:
            solver = get_solver()
            # print(f'Check for precision: {Not(Implies(unified_invs, tinv))}')
            solver.add(Not(Implies(unified_invs, tinv)))
            result = timed_check(solver)['result']
            if result == unsat:
                found_true_invs.append(tinv)

    correct_predicted_invs = []
    solver = get_solver()
    solver.add(unified_true_invs)
    result = timed_check(solver)['result']
    # assert result == sat, (result, unified_true_invs, solver.get_model())
    for inv in loop_invs_z3:
        solver = get_solver()
        # print(f'Check for recall: {Not(Implies(unified_true_invs, inv))}')
        solver.add(Not(Implies(unified_true_invs, inv)))
        result = timed_check(solver)['result']
        if result == unsat:
            correct_predicted_invs.append(inv)
    precision = len(correct_predicted_invs) / len(loop_invs_z3) if len(loop_invs_z3) > 0 else 0
    recall = len(found_true_invs) / len(true_loop_invs_z3) if len(true_loop_invs_z3) > 0 else 0

    assert recall <= 1, (recall, loop_invs, true_loop_invs,)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'simplified_loop_invs': simplied_loop_invs,
        'simplified_true_loop_invs': simplied_true_loop_invs,
        'raw_loop_invs': loop_invs,
        'raw_true_loop_invs': true_loop_invs,
        'found_true_loop_invs': [str(inv).replace('\n', ' ') for inv in found_true_invs],
        'correct_predicted_invs': [str(inv).replace('\n', ' ') for inv in correct_predicted_invs],
        'missing_true_loop_invs': [str(inv).replace('\n', ' ') for inv in true_loop_invs_z3 if inv not in found_true_invs],
    }

if __name__ == '__main__':
    # metrics = check_z3_problem('cohencu', 1, ['z == 6 * n + 6', 'y == 3 * n * n + 3 * n + 1', 'x == n * n * n', 'n <= a + 1'])
    metrics = check_z3_problem('egcd2', 1, ['Bézout_identity = z3.And(p*x + q*y == a, r*x + s*y == b)', 'z3.And(p*x + q*y == a, r*x + s*y == b)', 'multiples = z3.And(z3.Not(z3.Not(z3.Exists([z3.k], a == x*z3.k))), z3.Not(z3.Not(z3.Exists([z3.k], b == y*z3.k))))', 'a >= b', 'z3.And(z3.Not(z3.Not(z3.Exists([z3.k], a == x*z3.k))), z3.Not(z3.Not(z3.Exists([z3.k], b == y*z3.k))))', 'order = a >= b'])
    print(metrics)
