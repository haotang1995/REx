import sys
from z3 import *

#print(sys.argv)

smt = sys.argv[-1]

solver = Solver()
solver.from_string(smt)
result = solver.check()


print(result)

if result == sat:
    print(str(solver.model()))

