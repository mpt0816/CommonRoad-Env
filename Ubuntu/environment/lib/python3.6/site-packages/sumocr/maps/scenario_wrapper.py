from abc import ABCMeta
from commonroad.scenario.lanelet import LaneletNetwork

__author__ = "Moritz Klischat"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["ZIM Projekt ZF4086007BZ8"]
__version__ = "2021.5"
__maintainer__ = "Moritz Klischat"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"

from commonroad.scenario.scenario import Scenario


class AbstractScenarioWrapper(metaclass=ABCMeta):
    # these attributes are abstract and need to be defined by the inheriting subclass
    sumo_cfg_file = ""
    planning_problem_set = None
    _initial_scenario = None

    @property
    def lanelet_network(self) -> LaneletNetwork:
        return self._lanelet_network

    @lanelet_network.setter
    def lanelet_network(self, _):
        raise RuntimeError("lanelet_network cannot be set, set initial_scenario instead")

    @property
    def initial_scenario(self) -> Scenario:
        return self._initial_scenario

    @initial_scenario.setter
    def initial_scenario(self, initial_scenario):
        self._initial_scenario = initial_scenario
        self._lanelet_network = initial_scenario.lanelet_network

    def create_minimal_scenario(self) -> Scenario:
        sc = Scenario(dt=self.initial_scenario.dt, scenario_id=self.initial_scenario.scenario_id)
        sc.lanelet_network = self.initial_scenario.lanelet_network
        return sc

    def create_full_meta_scenario(self) -> Scenario:
        sc = self.create_minimal_scenario()
        sc.author = self.initial_scenario.author
        sc.tags = self.initial_scenario.tags
        sc.affiliation = self.initial_scenario.affiliation
        sc.source = self.initial_scenario.source
        sc.location = self.initial_scenario.location
        return sc

