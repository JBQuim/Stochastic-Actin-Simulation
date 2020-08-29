import stochastic.tools as sto
import matplotlib.pyplot as plt
import timeit


sys = sto.System(['X'], [250])
reactions = [
    sto.ElementaryStep([0, 0], [1], 0.015),
    sto.NthOrderReaction([3], [-1], 0.0001/6),
    sto.ElementaryStep([], [1], 200),
    sto.ElementaryStep([0], [-1], 3.5)
]
sim = sto.Simulation(reactions, sys, max_it=1000)
sim.simulate()
plt.plot(sim.times, sim.history)

plt.legend(sim.system.names)
plt.show()