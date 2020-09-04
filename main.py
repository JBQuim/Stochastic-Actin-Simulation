import stochastic.tools as sto
import stochastic.filaments as fsto
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

sys = fsto.FilamentSystem(['A', 'Active strands'], [200, 20], ['Empty', 'Full', 'Aged'], max_length=80, min_length=3, max_filaments=50)
sys.populate([[2]*5 + [1]*5]*20)
reactions = [
    fsto.NucleateFilament([1]*5, [1, 0], [-5, 1], 3e-4),
    fsto.Shortening('+', 2, [1, 0], 8e-3, removal_change=[3, -1]),
    fsto.Shortening('-', 2, [1, 0], 8e-2, removal_change=[3, -1]),
    fsto.ChangeUnitState(1, 2, [0, 0], 2e-2),
    fsto.Grow('-', [1, 2], 1, [1, 0], [-1, 0], 6e-5),
    fsto.Grow('+', [1, 2], 1, [1, 0], [-1, 0], 6e-4)
]
record_config = {
    'concs': (10, 'steps'),
    'size': (10, 'steps'),
    'filaments': (10, 'time'),
    'positions': (1, 'steps'),
    '_time_length': 3000,
    '_dynamic': True
}
sim = sto.Simulation(reactions, sys, record_config, max_it=80000)
sim.simulate()

plt.figure()
plt.plot(sim.times[1::10], sim.history_concs[1:] / sim.history_size[1:, np.newaxis])
plt.legend(sim.system.names)
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.show()

plt.figure()
positions, times = fsto.process_positions(sim.history_positions, sim.times)
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
    im = plt.imshow(frame, animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=1)
plt.show()
