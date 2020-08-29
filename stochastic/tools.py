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
        :param species_names: a list of strings of the names of all the species.
        :param initial_concs: the initial amount of every species in the system
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
    def __init__(self, reactions, system, max_it=100):
        """
        :param reactions: a list of reaction instances, following the BaseReaction API
        :param system: a system instance holding the state of the simulation
        :param max_it: the max number of iterations before stopping the simulation
        """
        self.reactions = reactions
        self.reaction_count = len(reactions)
        self.system = system
        self.t = 0
        self.max_it = max_it
        self.history = np.full((self.max_it + 1, len(system.concs)), np.nan)
        self.times = np.full(self.max_it + 1, np.nan)

    def _record(self, iteration):
        """
        Saves the current state to the simulation for later retrieval. The state of system.concs is saved to self.history
        and self.times. Behaviour can be extended by subclasses of Simulation to save more information.
        :param iteration: A number describing the current iterations performed.
        :return: None
        """
        self.history[iteration] = self.system.concs
        self.times[iteration] = self.t

    def _step(self):
        """
        Step the simulation forward according to Gillespie's algorithm.
        :return: True if the total propensities is 0, otherwise False. Returning True will end the simulation.
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

    def simulate(self, end_condition=lambda x: False):
        """
        Modifies the system by repeatedly calling self._step(). Simulation ends when either of self._step() or
        end_condition(self) return True.
        :param end_condition: Function that takes in a simulation instance and determines if it should end. If it returns
        True, the simulation ends.
        :return: None
        """
        self._record(0)
        for iteration in range(self.max_it):
            if end_condition(self):
                break

            end = self._step()
            self._record(iteration + 1)
            if end:
                break


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
    Elementary reaction step of zeroth, first or second order. Faster implementation of NthOrderReaction but restricted to
    smaller reaction orders.
    """
    def __init__(self, reactants, changes, rate_constant):
        """
        :param reactants: a list of the IDs of the reactants involved in the reaction. If the reaction involves a reactant
        twice, it should be repeated in the list.
        :param changes: a list of the change to every species when the reaction takes place
        :param rate_constant: the kinetic rate constant.
        """
        self.order = len(reactants)
        self.reactants = reactants
        self.changes = changes
        self.rate_constant = rate_constant

        self._check_valid()

    def _check_valid(self):
        assert self.order in [0, 1, 2], 'Order of reaction has to be 0, 1 or 2'

    def get_propensity(self, system):
        combinations = np.product(system.concs[self.reactants])
        if self.order == 2:
            if self.reactants[0] == self.reactants[1]:
                # if reaction is first order with respect to same reactant. The permutations are N(N-1). Because N^2 has
                # been calculated. An excess of N combinations have been counted.
                combinations -= system.concs[self.reactants[0]]

        return self.rate_constant * combinations * system.size ** (1 - self.order)

    def modify_system(self, system):
        system.concs += self.changes
        system.concs[system.concs < 0] = 0


class NthOrderReaction(ElementaryStep):
    """
    Elementary reaction step of any order.
    """
    def __init__(self, *args):
        """
        :param args: same as those of ElementaryStep but the reactants argument is now a list with the order of the
        reaction with respect to every species.
        """
        super().__init__(*args)
        self.order = np.sum(self.reactants)

    def _check_valid(self):
        assert len(self.reactants) == len(
            self.changes), 'List of orders with respect to every species must be the same length as the changes in species'

    def get_propensity(self, system):
        combinations = perm(system.concs, self.reactants).prod()
        return self.rate_constant * combinations * system.size ** (1 - self.order.astype(float))
