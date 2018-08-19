import time
from common_c import simulation as simulation_c

t0  = time.clock()
state  = initial_state()
print_state(state)
print()
for _ in range(35):
    state = random_next_state(state)
    if state:
        print_state(state)
        print()
    else:
        break

print(time.clock() - t0)

t0  = time.clock()
results,counts = simulation(state)
t1  = time.clock() - t0
print(t1)

t0  = time.clock()
results2,counts2 = simulation_c(state)
t2  = time.clock() - t0
print(t2)

# print(*results,sep='\n')
print(t1/t2)
print(all([x.equals(y) for x,y in zip(results,results2)]))
