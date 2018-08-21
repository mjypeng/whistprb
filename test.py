import time, pstats, cProfile, io
from common import *

t0  = time.clock()
state  = initial_state()
print_state(state)
print()
for _ in range(24):
    state = next_state(state)
    if state:
        print_state(state)
        print()
    else:
        break

print(time.clock() - t0)

clear_memoized_states()
cProfile.runctx("p,r,c=simulation(state,goals='min',terminal='round_end')",globals(),locals(),'sim_stats.prf')
s = pstats.Stats('sim_stats.prf')
s.strip_dirs().sort_stats('time').print_stats()

# t0  = time.clock()
# r,c = simulation(state,full=True)
# t1  = time.clock() - t0
# print(t1)

# t0  = time.clock()
# p,r,c = simulation(state,goals='min',terminal='round_end')
# t2  = time.clock() - t0
# print(p)
# print(t2)

# t0  = time.clock()
# results2,counts2 = simulation_c(state)
# t2  = time.clock() - t0
# print(t2)

# print(*results,sep='\n')
# print(t1/t2)
# print(all([x.equals(y) for x,y in zip(results,results2)]))
