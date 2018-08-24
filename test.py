import time, pstats, cProfile, io
from line_profiler import LineProfiler
from common import *

# Case Studies:

t0  = time.clock()
state  = initial_state()
print_state(state)
print()
for _ in range(32):
    state = next_state(state)
    if state:
        print_state(state)
        print()
    else:
        break

print(time.clock() - t0)

# reset_memoized_states()
cProfile.runctx("r=simulation(state,goals='min',terminal='round_end',return_path=True)",globals(),locals(),'sim_stats.prf')
s = pstats.Stats('sim_stats.prf')
print("return_path=True")
s.strip_dirs().sort_stats('time').print_stats()
print()

print_state(state)
print(pd.DataFrame([(card_to_console(x[0]),x[1]) for x in r],columns=('play','scores')).drop_duplicates())

reset_memoized_states()
lp  = LineProfiler(simulation_step)
lp.runctx("r=simulation(state,goals='min',terminal='round_end',return_path=True)",globals(),locals())
lp.print_stats()

# reset_memoized_states()
# cProfile.runctx("r2=simulation(state,goals='min',terminal='round_end',return_path=False)",globals(),locals(),'sim_stats.prf')
# s = pstats.Stats('sim_stats.prf')
# print("return_path=False")
# s.strip_dirs().sort_stats('time').print_stats()
# print()

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
