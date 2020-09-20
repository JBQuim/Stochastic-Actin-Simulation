import stochastic.tools as sto
import stochastic.filaments as fsto
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

sys = fsto.FilamentSystem(['Actin', 'Active strands'],
                          [4000, 0], ['Empty', 'New', 'Aged'],
                          max_length=100, min_length=3, max_filaments=60, shrink_factor=1.1)
# sys.populate([[1] * 20] * 45)

reactions = [
    fsto.NucleateFilament([1] * 5, [2, 0], [-5, 1], 6e-9),
    fsto.Shortening('+', 2, [1, 0], 1.8e-1, removal_change=[3, -1]),
    fsto.Shortening('-', 2, [1, 0], 1.2e-2, removal_change=[3, -1]),
    fsto.Shortening('+', 1, [1, 0], 9e-3, removal_change=[3, -1]),
    fsto.Shortening('-', 1, [1, 0], 6e-4, removal_change=[3, -1]),
    fsto.ChangeUnitState(1, 2, [0, 0], 1.5e-3),
    fsto.Grow('-', [1, 2], 1, [1, 0], [-1, 0], 4e-7),
    fsto.Grow('+', [1, 2], 1, [1, 0], [-1, 0], 6e-6)
]

sampling_rate = 10, 30
record_config = {
    'concs': (sampling_rate[0], 'steps'),
    'size': (sampling_rate[0], 'steps'),
    'filaments': (sampling_rate[1], 'time'),
    'positions': (1, 'steps'),
    '_time_length': 8000,
    '_dynamic': True
}
sim = sto.Simulation(reactions, sys, record_config, max_it=80000)
sim.simulate()
# fsto.preview_graphs(sim.times, sim.history_concs, sim.history_size, sampling_rate, sim.system.names,
#                    sim.history_filaments, sim.history_positions)
# fsto.produce_ani_frames('Data/ani', sim.times, sim.history_concs, sim.history_size, sampling_rate,
# sim.system.names, sim.history_filaments, sim.history_positions)
# np.savez('Data/actin', sim.times,
# sim.history_concs, sim.history_size, sampling_rate, sim.system.names, sim.history_filaments, sim.history_positions)