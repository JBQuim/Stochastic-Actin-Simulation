import stochastic.tools as sto
import stochastic.filaments as fsto
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

with np.load('Data/actin.npz', allow_pickle=False) as arrays:
    times, history_concs, history_size, sampling_rate, names, history_filaments, history_positions = arrays.values()
fsto.produce_ani_frames('Data/ani',
                        times, history_concs, history_size, sampling_rate, names, history_filaments, history_positions)
