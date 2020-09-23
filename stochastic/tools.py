from abc import ABC, abstractmethod
import numpy as np
from random import choices
from scipy.special import perm


class System:
    """
    Holds the state of the system. This includes the amount of each species and the species names.
    """

    def __init__(self, species_names, initial_concs, size=1):
        """
        :param species_names: list of length S, holding strings of the names of all the species.
        :param initial_concs: list of length S, holding the initial amount of every species in the system
        :param size: the system size. Used in converting the rate constants to stochastic rate constants
        """
        assert len(species_names) == len(initial_concs), 'Must have the same amount of species labels and amounts'

        self.names = np.array(species_names)
        self.concs = np.array(initial_concs)
        self.size = size

    def __repr__(self):
        lines = ["{} = {}".format(name, conc) for name, conc in zip(self.names, self.concs)]
        return '\n'.join(lines)


class Simulation:
    """
    Implements the logic behind the stochastic simulation of the evolution of the system.
    """

    def __init__(self, reactions, system, record_config, max_it=100):
        """
        :param reactions: a list of length R, holding reaction instances, following the BaseReaction API
        :param system: a system instance holding the state of the simulation
        :param record_config: a dictionary outlining the settings of _record. Keys are variables names that must be saved
        from system as the simulation is run. They are saved to the simulation attribute history_[var name]. See _record
        for more details.
        :param max_it: the max number of iterations before stopping the simulation
        """
        self.reactions = reactions
        self.reaction_count = len(reactions)
        self.system = system
        self.t = 0
        self.max_it = max_it

        self.record_time_length = record_config.pop('_time_length', 5)
        self.record_dynamic = record_config.pop('_dynamic', False)
        self.record_config = record_config
        self.history = dict()

        self.times = np.full(self.max_it + 1, np.nan)
        for var_name, (period, mode) in record_config.items():
            data_shape = np.shape(getattr(system, var_name))
            if mode == 'steps':
                assert isinstance(period, int), 'Period for step must be an integer'
                history_shape = (int(self.max_it / period) + 1, *data_shape)
            elif mode == 'time':
                history_shape = (int(self.record_time_length / period) + 1, *data_shape)
            else:
                raise NotImplementedError('Mode {} not understood. Mode must be one of steps or time.'.format(mode))
            self.history[var_name] = np.full(history_shape, np.nan)

    def _record(self, iteration, latest_times):
        """
        Saves the current state to the simulation for later retrieval. The variables saved from self.system are the keys
        of the dictionary provided when initializing the simulation. They are saved to self.hystory_[var name].

        The value of each entry in the config dictionary should be a tuple, where the first element is the period of the
        recording and the second whether this is in time or steps. e.g. (2, 'steps') is every two steps and (0.2, 'time')
        is every 0.2 units of time. Additionally, the value for _time_length is the length of time for which the time
        parameters are recorded. If _dynamic is True then this is taken only as estimation of the length of the simulation,
        but data is recorded for the whole duration.
        :param iteration: The number of iterations performed so far.
        :latest_times: A dictionary of the time at which each variable was last saved.
        :return: updated dictionary of latest_times
        """

        # initialize the latest_times dict
        if iteration == 0:
            latest_times = dict({})
            for var_name, (period, mode) in self.record_config.items():
                if mode == 'time':
                    latest_times[var_name] = 0

        self.times[iteration] = self.t
        for var_name, (period, mode) in self.record_config.items():
            data = getattr(self.system, var_name)
            history_data = self.history[var_name]
            if mode == 'steps':
                if iteration % period == 0:
                    index = iteration // period
                    history_data[index] = data
            elif mode == 'time':
                samples_taken = int(latest_times[var_name] / period)
                new_total_samples = int(self.t / period)
                if samples_taken == new_total_samples:  # if there is no need to record new samples
                    continue
                elif new_total_samples >= len(history_data):  # if new samples exceed the memory already allocated
                    if self.record_dynamic:
                        print('Resizing memory allocation for history_{}'.format(var_name))
                        length_estimate = int(new_total_samples * max(1.2, self.max_it / iteration))
                        # add padding only along the first axis, at the end
                        padding_widths = [(0, length_estimate - samples_taken)] + [(0, 0)] * (history_data.ndim - 1)
                        history_data = np.pad(history_data, padding_widths, 'constant', constant_values=np.nan)
                    else:
                        continue
                index = np.arange(samples_taken, new_total_samples, 1, dtype=np.int32)
                history_data[index] = data
                latest_times[var_name] = self.t
            self.history[var_name] = history_data

        return latest_times

    def _step(self):
        """
        Step the simulation forward according to Gillespie's algorithm.
        :return: True if the total of the propensities is 0, otherwise False. Returning True will end the simulation.
        """

        propensities = np.array([reaction.get_propensity(self.system) for reaction in self.reactions], dtype=np.float_)
        total_prop = propensities.sum()

        if total_prop == 0:
            return True

        t_react = np.random.exponential(1 / total_prop)
        reaction_num = choices(range(self.reaction_count), weights=propensities)[0]
        self.reactions[reaction_num].modify_system(self.system)
        self.t += t_react

        return False

    def simulate(self, repeat_func=lambda x: False):
        """
        Modifies the system by repeatedly calling self._step(). Simulation ends when self._step() or repeat_func(self)
         return True or when maximum iterations are reached.
        :param repeat_func: Function that is ran every step. It takes the simulation instance as argument. Simulation ends if it returns True.
        :return: None
        """
        record_cache = self._record(0, None)
        for iteration in range(self.max_it):
            if repeat_func(self):
                break

            end = self._step()
            record_cache = self._record(iteration + 1, record_cache)
            if end:
                break

            print(iteration)


class BaseReaction(ABC):
    """
    Template for all reactions. Each reaction class must implement the two functions get_propensity(system) and
    modify_system(system). Reaction instances are passed to the simulator in a list.
    """

    @abstractmethod
    def get_propensity(self, system):
        """
        Calculates the propensity of the reaction given the system.
        :param system: an instance of a system class, with all the relevant concentrations to the reaction.
        :return: None. Children of this class should overwrite this function and have it return a float of the reaction's
        propensity.
        """
        pass

    @abstractmethod
    def modify_system(self, system):
        """
        :param system: an instance of a system class. The function will modify the instance according to the particular
        reaction.
        :return: None
        """
        pass


class ElementaryStep(BaseReaction):
    """
    Elementary reaction step of any order.
    """

    def __init__(self, reactants, changes, rate_constant):
        """
        :param reactants: list of length S, holding the order of the reaction with respect to every species
        :param changes: list of length S, holding the change to every species when the reaction takes place
        :param rate_constant: the kinetic rate constant.
        """
        self.reactants = np.array(reactants)
        self.changes = changes
        self.rate_constant = rate_constant
        self.order = self.reactants.sum()

        self._check_valid()

    def _check_valid(self):
        assert len(self.reactants) == len(
            self.changes), 'List of orders with respect to every species must be the same length as the changes in ' \
                           'species '

    def get_propensity(self, system):
        combinations = perm(system.concs, self.reactants).prod()
        # make sure there are enough reactants for reaction
        if not (system.concs >= self.reactants).all() or (system.concs + self.changes < 0).any():
            return 0
        else:
            return self.rate_constant * combinations * system.size ** (1 - self.order.astype(float))

    def modify_system(self, system):
        system.concs += self.changes
