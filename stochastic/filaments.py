import numpy as np
import stochastic.tools as sto
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from random import choice
from scipy.special import perm


class FilamentSystem(sto.System):
    """
    Extends the system class, adding functionality to record the state of filaments and their relative positions.
    """

    def __init__(self, species_names, initial_concs, state_names, size=1, max_filaments=50, max_length=100,
                 min_length=4, shrink_factor=3):
        """
        :param species_names: list of length S, holding strings of the names of all the species.
        :param initial_concs: list of length S, holding the initial amount of every species in the system
        :param state_names: list of length ST, holding strings of the names of all the filament states
        :param size: the system size. Used in converting the rate constants to stochastic rate constants.
        :param max_filaments: the maximum number of filaments at any one time
        :param max_length: the maximum length of a filament
        :param min_length: minimum length of a filament before it is removed
        :param shrink_factor: when there are too many filaments, the observed volume will be divided by this number.
        """
        super().__init__(species_names, initial_concs, size=size)
        self.state_names = state_names
        self.max_filaments = max_filaments
        self.max_length = max_length
        self.min_length = min_length
        self.shrink_factor = shrink_factor

        self.filaments = np.zeros((max_filaments, max_length))
        self.positions = np.full(max_filaments, np.nan)

    def clear(self):
        """
        Resets the system's filaments
        :return: None
        """
        self.filaments = np.zeros((self.max_filaments, self.max_length))
        self.positions = np.full(self.max_filaments, np.nan)

    def populate(self, filaments, positions=None):
        """
        Populates the system with the given filaments.
        :params filaments: a list of lists with dimension N x L with every entry corresponding to the state
         of a filament at a gien point. N must be less than max_filaments and L must be less than max_length.
        :params positions: a list of length N, holding the starting positions of the strands.
        :return: None
        """
        filaments = np.array(filaments)
        N, L = filaments.shape
        ML = self.max_length
        assert not np.any(self.filaments), 'Filaments must be 0 before populating. Consider using clear()'
        assert np.isnan(
            self.positions).all(), 'Filament positions must be nan before populating. Consider using clear()'
        assert N <= self.max_filaments, 'Too many filaments. Allowed maximum is {}, {} provided'.format(
            self.max_filaments, N)
        assert L <= ML, 'Filaments too long. Allowed maximum is {}, {} provided'.format(ML, L)
        assert np.all(filaments >= 0) and np.all(filaments < len(self.state_names)), \
            'No state corresponding to the number provided. All numbers must integers between 0 and {}'.format(
                len(self.state_names) - 1)

        # center filament in array
        self.filaments[:N, (ML - L) // 2:(ML + L) // 2] = filaments

        if positions is None:
            self.positions[:N] = 0
        else:
            assert N == len(positions), 'Number of positions given must match number of filaments given'
            self.positions[:N] = positions

    def shrink(self):
        """
        Modifies volume when there are too many strands.
        """
        self.size /= self.shrink_factor
        self.concs = (self.concs / self.shrink_factor).astype(self.concs.dtype)
        mask = np.random.rand(self.max_filaments) > (1 / self.shrink_factor)
        self.filaments[mask] = 0
        self.positions[mask] = np.nan


class NucleateFilament(sto.ElementaryStep):
    """
    Create a filament.
    """

    def __init__(self, smallest_filament, reactants, changes, rate_constant, start_pos=0):
        """
        :param smallest_filament: list with the states of the filament when it is created
        :param reactants: list of length S, holding the order of the reaction with respect to every species
        :param changes: list of length S, holding the change to every species when the reaction takes place
        :param rate_constant: rate constant for reaction
        :param start_pos: starting position of the filament
        """
        super().__init__(reactants, changes, rate_constant)
        self.smallest = np.array(smallest_filament)
        self.start_pos = start_pos

    def modify_system(self, fsystem):
        filament_present = fsystem.filaments.any(axis=1)
        N = filament_present.sum()
        empty_slot = filament_present.argmin()
        ML = fsystem.max_length
        L = len(self.smallest)

        fsystem.filaments[empty_slot, (ML - L) // 2:(ML + L) // 2] = self.smallest
        fsystem.positions[empty_slot] = self.start_pos

        fsystem.concs += self.changes

        if N + 1 == fsystem.max_filaments:
            fsystem.shrink()


class Shortening(sto.BaseReaction):
    """
    Remove a subunit from the end of one of the filaments.
    """

    def __init__(self, end, state, changes, rate_constant, removal_change=None):
        """
        :param end: '+' or '-' to remove from the right or left, respectively.
        :param state: integer corresponding to the state of the tip needed for it to lose a unit
        :param changes: list of length S, holding the changes to every species when a unit is removed
        :param rate_constant: rate constant for reaction
        :param removal_change: list of length S, holding the changes to every species when a whole strand is removed
        """
        self.end = end
        self.state = state
        self.changes = np.array(changes)
        self.removal_change = self.changes if removal_change is None else np.array(removal_change)
        self.rate_constant = rate_constant
        self.cache = None

        self._check_valid()

    def _check_valid(self):
        assert self.end in ['+', '-'], 'The value of end given must be + or -.'

    def get_propensity(self, fsystem):
        if (fsystem.concs + self.changes < 0).any():
            return 0

        if self.end == '+':
            arr = np.flip(fsystem.filaments, axis=1)
        else:
            arr = fsystem.filaments

        keys = np.argmax(arr != 0, axis=1)
        ends = arr[np.arange(arr.shape[0]), keys]
        correct_ends = ends == self.state

        if (fsystem.concs + self.removal_change < 0).any():
            lengths = (fsystem.filaments > 0).sum(axis=1)
            correct_ends *= lengths <= fsystem.min_length

        counts = np.sum(correct_ends)
        self.cache = correct_ends, keys
        return self.rate_constant * counts

    def modify_system(self, fsystem):
        correct_ends, keys = self.cache
        selected_strand_key = choice(correct_ends.nonzero()[0])

        length_selected = (fsystem.filaments[selected_strand_key] > 0).sum()
        if length_selected <= fsystem.min_length:
            fsystem.filaments[selected_strand_key] = np.zeros(fsystem.max_length)
            fsystem.concs += self.removal_change
            fsystem.positions[selected_strand_key] = np.nan
        else:
            if self.end == '+':
                tail_pos = fsystem.max_length - keys[selected_strand_key] - 1
            else:
                tail_pos = keys[selected_strand_key]
                fsystem.positions[selected_strand_key] += 1
            fsystem.filaments[selected_strand_key, tail_pos] = 0
            fsystem.concs += self.changes


class ChangeUnitState(sto.BaseReaction):
    """
    Change the state of a filament unit into a different one
    """

    def __init__(self, old_state, new_state, changes, rate_constant):
        """
        :param old_state: integer representing the reacting state
        :param new_state: integer representing the new state
        :param changes: list of length S, holding the changes to every species when the reaction occurs
        :param rate_constant: rate constant for reaction
        """
        self.old_state = old_state
        self.new_state = new_state
        self.changes = np.array(changes)
        self.rate_constant = rate_constant

    def get_propensity(self, fsystem):
        if (fsystem.concs + self.changes < 0).any():
            return 0
        else:
            counts = np.sum(fsystem.filaments == self.old_state)
            return self.rate_constant * counts

    def modify_system(self, fsystem):
        y_keys, x_keys = np.nonzero(fsystem.filaments == self.old_state)
        chosen_pos = choice(range(x_keys.shape[0]))
        fsystem.filaments[y_keys[chosen_pos], x_keys[chosen_pos]] = self.new_state

        fsystem.concs += self.changes


class Grow(sto.BaseReaction):
    def __init__(self, end, old_states, new_state, reactants, changes, rate_constant):
        self.end = end
        self.old_states = np.array(old_states)
        self.new_state = new_state
        self.reactants = np.array(reactants)
        self.changes = np.array(changes)
        self.rate_constant = rate_constant
        self.order = self.reactants.sum() + 1
        self.cache = None

        self._check_valid()

    def _check_valid(self):
        assert self.end in ['+', '-'], 'The value of end given must be + or -.'

    def get_propensity(self, fsystem):
        if (fsystem.concs + self.changes < 0).any():
            return 0

        if self.end == '+':
            arr = np.flip(fsystem.filaments, axis=1)
        else:
            arr = fsystem.filaments

        keys = np.argmax(arr != 0, axis=1)
        ends = arr[np.arange(arr.shape[0]), keys]
        lengths = (arr != 0).sum(axis=1)
        correct_ends = np.isin(ends, self.old_states) * (lengths < fsystem.max_length)

        counts = np.sum(correct_ends)
        self.cache = correct_ends, keys

        combinations = perm(fsystem.concs, self.reactants).prod()
        if (fsystem.concs >= self.reactants).all():
            return self.rate_constant * counts * combinations * fsystem.size ** (1 - self.order.astype(float))
        else:
            return 0

    def modify_system(self, fsystem):
        correct_ends, keys = self.cache
        selected_strand_key = choice(correct_ends.nonzero()[0])

        shift = 0
        if self.end == '+':
            tail_pos = fsystem.max_length - keys[selected_strand_key]
            if tail_pos == fsystem.max_length:
                L = np.sum(fsystem.filaments[selected_strand_key] != 0)
                shift = int((L - fsystem.max_length - 1) // 2)
        else:
            fsystem.positions[selected_strand_key] -= 1
            tail_pos = keys[selected_strand_key] - 1
            if tail_pos == -1:
                L = np.sum(fsystem.filaments[selected_strand_key] != 0)
                shift = int((fsystem.max_length - L + 1) // 2)

        fsystem.filaments[selected_strand_key] = np.roll(fsystem.filaments[selected_strand_key], shift)
        fsystem.filaments[selected_strand_key, tail_pos + shift] = self.new_state
        fsystem.concs += self.changes


def process_positions(positions, times):
    """
    Takes an array of histories of filament positions at every filament slot and outputs an array with the per filament
    trajectory. No filament at that slot requires it to be np.nan. At lower sampling frequencies a filament might not
    be detected as disappearing if it is quickly replaced.
    :param positions: a 2D array with the first dimension being time and the second dimension being filament slot.
    :param times: a 1D array with each element being the time at which the current position in the positions array was held
    :return: a list of trajectories for every filament.
    """
    pos_histories = []
    time_histories = []
    for slot_history in positions.T:
        nan_bools = np.isnan(slot_history)

        ends = np.full(nan_bools.shape, ~nan_bools[-1])
        ends[:-1] = nan_bools[1:] * (~nan_bools[:-1])

        starts = np.full(nan_bools.shape, ~nan_bools[0])
        starts[1:] = (~nan_bools[1:]) * nan_bools[:-1]

        start_pos = starts.nonzero()[0]
        end_pos = ends.nonzero()[0]

        for s, e in zip(start_pos, end_pos):
            pos_histories.append(slot_history[s:e + 1])
            time_histories.append(times[s:e + 1])
    return pos_histories, time_histories


def update_alphas(time, lines, alpha1, alpha2):
    for line in lines:
        xdata = line.get_xdata(orig=True)
        start_time, end_time = xdata[0], xdata[-1]
        alpha = alpha1 if start_time <= time <= end_time else alpha2
        line.set_alpha(alpha)


def produce_ani_frames(path, history_times, history_concs, history_size, sampling_rate, names, history_filaments,
                       history_positions):
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(5, 4)

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[1, :2])
    ax3 = fig.add_subplot(gs[2:, :2])
    ax4 = fig.add_subplot(gs[:, 2:])

    ax1.plot(history_times[::sampling_rate[0]], history_concs[:, 0] / history_size, lw=0.8)
    ax2.plot(history_times[::sampling_rate[0]], history_concs[:, 1] / history_size, lw=0.8)
    ax1.set_ylabel('Concentration')
    ax2.set_ylabel('Concentration')
    ax1.set_title(names[0])
    ax2.set_title(names[1])

    positions, times = process_positions(history_positions, history_times)
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    for pos, t in zip(positions, times):
        ax3.plot(t, pos, color=color, lw=0.8)
    ax3.set_ylabel('Displacement from origin')
    ax3.set_xlabel('Time')
    ax3.set_title('Strand positions')
    update_alphas(0, ax3.lines, 0.6, 0.2)

    ims = []
    ax1.axvline(0.05, color='black', alpha=0.5)
    ax2.axvline(0.05, color='black', alpha=0.5)
    ax3.axvline(0.05, color='black', alpha=0.5)
    coords = np.array([[0.05, 0], [0, 1]])
    for i in range(1228, len(history_filaments)):
        print(i)
        frame = history_filaments[i]
        time = (i + 1) * sampling_rate[1]
        if np.isnan(frame).all():
            break
        ax4.imshow(frame.T[::-1], animated=True)
        coords[0] = time
        ax1.lines[-1].set_data(coords)
        ax2.lines[-1].set_data(coords)
        ax3.lines[-1].set_data(coords)
        plt.savefig(path + str(i))
        plt.sca(ax4)
        plt.cla()

        update_alphas(time, ax3.lines[:-1], 0.6, 0.15)
        # ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=25)

    writer = animation.PillowWriter(fps=30)
    # ani.save('myAnimation2.gif', writer=writer)
    print("done")

    plt.figure()
    plt.plot(history_times[::sampling_rate[0]], history_size)
    plt.show()


def preview_graphs(history_times, history_concs, history_size, sampling_rate, names, history_filaments,
                   history_positions):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(history_times[::sampling_rate[0]], history_concs[:, 0] / history_size, lw=0.8)
    ax2.plot(history_times[::sampling_rate[0]], history_concs[:, 1] / history_size, lw=0.8)
    ax1.set_ylabel('Concentration')
    ax2.set_ylabel('Concentration')
    plt.show()

    plt.figure()
    positions, times = process_positions(history_positions, history_times)
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    for pos, t in zip(positions, times):
        plt.plot(t, pos, color=color, alpha=0.2)
    plt.xlabel('Displacement from origin')
    plt.ylabel('Time')
    plt.show()

    plt.figure()
    plt.plot(history_times[::sampling_rate[0]], history_size)

    fig = plt.figure()
    ims = []
    for frame in history_filaments:
        print(len(ims))
        if np.isnan(frame).all():
            break
        im = plt.imshow(frame.T[::-1], animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50)
    plt.show()
