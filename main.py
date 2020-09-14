import stochastic.tools as sto
import stochastic.filaments as fsto
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.optimize import fsolve


def equilibrate(simulation):
    k1 = 35.7e6 / 1.63e8 / omega
    # k2 = 2.15e6 / 1.3e3 / omega
    c0 = simulation.system.concs[0]

    estimate = np.array([c0, 0])

    def func(vector):
        monomer, dimer = vector[0], vector[1]
        a = k1 * monomer ** 2 - dimer
        b = monomer + 2 * dimer - c0
        return np.array([a, b])

    if simulation.system.concs[0] / simulation.system.size > 0.3e-6 and simulation.t > 300:
        simulation.system.concs[0] = 0.3e-6 * simulation.system.size

    roots = fsolve(func, estimate)
    simulation.system.concs[2] = roots[1].astype(np.int64)
    simulation.system.concs[1] = c0 - roots[1].astype(np.int64)

    return False


omega = 1e-9 * 6.2206e23
amount_initial = (np.array([3e-6 * omega, 0, 0, 0, 0])).astype(np.int64)

sys = fsto.FilamentSystem(['Total actin', 'Monomer', 'Dimer', 'Trimer', 'Active strands'],
                          amount_initial, ['Empty', 'ATP', 'ADP+P', 'ADP'],
                          max_length=500, min_length=4, max_filaments=100, size=omega, shrink_factor=1.2)
# sys.populate([[2]*15 + [1]*5]*200)

reactions = [
    fsto.NucleateFilament([1] * 4, [0, 1, 0, 1, 0], [-3, 0, 0, -1, 1], 11.6e6 + 1.3e6),
    fsto.Shortening('+', 1, [1, 0, 0, 0, 0], 1.4, removal_change=[3, 0, 0, 1, -1]),
    fsto.Shortening('-', 1, [1, 0, 0, 0, 0], 8e-1, removal_change=[3, 0, 0, 1, -1]),
    fsto.Shortening('+', 2, [1, 0, 0, 0, 0], 1.4, removal_change=[3, 0, 0, 1, -1]),
    fsto.Shortening('-', 2, [1, 0, 0, 0, 0], 8e-1, removal_change=[3, 0, 0, 1, -1]),
    fsto.Shortening('+', 3, [1, 0, 0, 0, 0], 7.2, removal_change=[3, 0, 0, 1, -1]),
    fsto.Shortening('-', 3, [1, 0, 0, 0, 0], 2.7e-1, removal_change=[3, 0, 0, 1, -1]),
    fsto.ChangeUnitState(1, 2, [0, 0, 0, 0, 0], 3e-1),
    fsto.ChangeUnitState(2, 3, [0, 0, 0, 0, 0], 4e-3),
    fsto.Grow('+', [1, 2, 3], 1, [0, 1, 0, 0, 0], [-1, 0, 0, 0, 0], 11e6),
    fsto.Grow('-', [1, 2, 3], 1, [0, 1, 0, 0, 0], [-1, 0, 0, 0, 0], 1.3e6),
    sto.ElementaryStep([0, 1, 1, 0, 0], [-2, 0, 0, 1, 0], 2.15e6),
    sto.ElementaryStep([0, 0, 0, 1, 0], [2, 0, 0, -1, 0], 1.3e3)
]

record_config = {
    'concs': (50, 'steps'),
    'size': (50, 'steps'),
    'filaments': (0.1, 'time'),
    'positions': (50, 'steps'),
    '_time_length': 300,
    '_dynamic': True
}
sim = sto.Simulation(reactions, sys, record_config, max_it=int(5e5))
sim.simulate(repeat_func=equilibrate)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(sim.times[::50], sim.history_concs.T[3] / sim.history_size)
ax2.plot(sim.times[::50], sim.history_concs.T[4] / sim.history_size)
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.show()

plt.figure()
positions, times = fsto.process_positions(sim.history_positions, sim.times[::50])
color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
for pos, t in zip(positions, times):
    plt.plot(pos, t, color=color, alpha=0.2)
plt.xlabel('Displacement from origin')
plt.ylabel('Time')
plt.show()

fig = plt.figure()
ims = []
for frame in sim.history_filaments:
    print(len(ims))
    if np.isnan(frame).all():
        break
    im = plt.imshow(frame.T, animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50)
plt.show()

plt.figure()
plt.plot(sim.times[::50], sim.history_size)
