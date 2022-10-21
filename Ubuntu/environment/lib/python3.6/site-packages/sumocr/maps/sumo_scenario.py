import logging
import os
import pathlib
import xml.etree.ElementTree as et

from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.common.file_reader import CommonRoadFileReader

from sumocr.interface.util import NetError
from sumocr.sumo_config import DefaultConfig
from sumocr.maps.scenario_wrapper import AbstractScenarioWrapper

__author__ = "Moritz Klischat"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["ZIM Projekt ZF4086007BZ8"]
__version__ = "2021.5"
__maintainer__ = "Moritz Klischat"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"


class ScenarioWrapper(AbstractScenarioWrapper):
    def __init__(self):
        self.scenario_name: str = ''
        self.net_file: str = ''
        self.cr_map_file: str = ''
        self.sumo_cfg_file = None
        self.ego_start_time: int = 0
        self.sumo_net = None
        self._lanelet_network: LaneletNetwork = None
        self._initial_scenario = None
        self.planning_problem_set = None
        self._route_planner = None

    def initialize(self,
                   scenario_name: str,
                   sumo_cfg_file: str,
                   cr_map_file: str,
                   ego_start_time: int = None) -> None:
        """
        Initializes the ScenarioWrapper.

        :param scenario_name: the name of the scenario
        :param sumo_cfg_file: the .sumocfg file
        :param cr_map_file: the commonroad map file
        :param ego_start_time: the start time of the ego vehicle

        """
        self.scenario_name = scenario_name
        self.sumo_cfg_file = sumo_cfg_file
        self.net_file = self._get_net_file(self.sumo_cfg_file)
        self.cr_map_file = cr_map_file
        self.ego_start_time = ego_start_time
        self.initial_scenario, self.planning_problem_set = CommonRoadFileReader(self.cr_map_file).open()
        if len(self.planning_problem_set.planning_problem_dict) == 0:
            self.planning_problem_set = None

    @classmethod
    def init_from_scenario(cls,
                           config: DefaultConfig,
                           scenario_path=str,
                           ego_start_time: int = None,
                           cr_map_file=None) -> 'ScenarioWrapper':
        """
        Initializes the ScenarioWrapper according to the given scenario_name/ego_start_time and returns the ScenarioWrapper.
        :param config: config file for the initialization, contain scenario_name.
        :param scenario_path: path to the scenario folder
        :param ego_start_time: the start time of the ego vehicle.
        :param cr_map_file: path to commonroad map, if not in scenario folder

        """
        assert isinstance(
            config,
            DefaultConfig), f'Expected type DefaultConfig, got {type(config)}'

        obj = cls()
        scenario_path = config.scenarios_path if config.scenarios_path is not None else scenario_path
        sumo_cfg_file = os.path.join(scenario_path,
                                     config.scenario_name + '.sumo.cfg')
        if cr_map_file is None:
            cr_map_file = os.path.join(scenario_path,
                                       config.scenario_name + '.cr.xml')

        obj.initialize(config.scenario_name, sumo_cfg_file, cr_map_file,
                       ego_start_time)
        return obj

    def _get_net_file(self, sumo_cfg_file: str) -> str:
        """
        Gets the net file configured in the cfg file.

        :param sumo_cfg_file: SUMO config file (.sumocfg)

        :return: net-file specified in the config file
        """
        if not os.path.isfile(sumo_cfg_file):
            raise ValueError(
                "File not found: {}. Maybe scenario name is incorrect.".format(
                    sumo_cfg_file))
        tree = et.parse(sumo_cfg_file)
        file_directory = os.path.dirname(sumo_cfg_file)
        # find net-file
        all_net_files = tree.findall('*/net-file')
        if len(all_net_files) != 1:
            raise NetError(len(all_net_files))
        return os.path.join(file_directory, all_net_files[0].attrib['value'])

    def get_rou_file(self) -> str:
        """
        Gets the net file configured in the cfg file.

        :param sumo_cfg_file: SUMO config file (.sumocfg)

        :return: net-file specified in the config file
        """
        if not os.path.isfile(self.sumo_cfg_file):
            raise ValueError(
                "File not found: {}. Maybe scenario name is incorrect.".format(
                    self.sumo_cfg_file))
        tree = et.parse(self.sumo_cfg_file).find("input")
        file_directory = os.path.dirname(self.sumo_cfg_file)
        # find net-file
        rou_files = tree.findall('route-files')
        if len(rou_files) > 1:
            for r in rou_files:
                "vehicle" in r.get("value")
                rou_file = r
        else:
            rou_file = rou_files[0]

        rou_path = os.path.join(file_directory, rou_file.attrib['value'])
        if not os.path.isfile(rou_path):
            return None
        return rou_path
