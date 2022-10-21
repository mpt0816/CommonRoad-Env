import inspect
import pickle
from collections import defaultdict
from enum import Enum

import grpc
import os
from grpc import ChannelCredentials
import logging
from functools import wraps
from functools import partial
from typing import Dict, Union, List, Tuple

from commonroad.scenario.scenario import Scenario, LaneletNetwork
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import State
from sumocr.interface.ego_vehicle import EgoVehicle
from sumocr.interface.sumo_simulation import SumoSimulation
from sumocr.maps.scenario_wrapper import AbstractScenarioWrapper
from sumocr.maps.sumo_scenario import ScenarioWrapper
from sumocr.sumo_config import DefaultConfig

from sumocr.sumo_docker.rpc.messaging import get_data_chunks, save_chunks_to_file, \
    unzip_file


__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["ZIM Projekt ZF4086007BZ8"]
__version__ = "0.7.4"
__maintainer__ = "Peter Kocsis"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Integration"


class SumoRPCClient:
    """RPC client between CommonRoad and SUMO. This class mirrors all public functions and attributes from
    the class :class:`~interface.sumo_simulation.SumoSimulation`.

    """
    __version__ = __version__
    __rpc_module = 'sumo_manager_server.sumo_interface'

    logger = logging.getLogger(__rpc_module)
    logger.setLevel(logging.INFO)

    class Decorators:
        @classmethod
        def log_call(cls, level=logging.DEBUG):
            """
            Logging decorator
            :param level: The level of the logging
            """

            def decorator(func):
                @wraps(func)
                def logged_call(*args, **kwargs):
                    SumoRPCClient.logger.log(level, func.__qualname__ + " called")
                    return func(*args, *kwargs)

                return logged_call

            return decorator

    def __init__(self, address: str):
        """Init empty object"""
        logging.basicConfig()
        self._address = address

        credentials = self.get_trusted_credentials()
        self.channel = grpc.secure_channel(self._address, credentials)
        self._request_provider = SumoRPCClient.RequestProvider(self.channel, self.__rpc_module)

        def not_implemented():
            raise   NotImplementedError()
        self._request = defaultdict(lambda: not_implemented)

        # ============================= GRPC REQUESTS ===============================
        self._create_helper_requests()
        if not self.check_connection():
            raise RuntimeError("GRPC server is unavailable!")

        self._create_property_requests()
        self._create_variable_requests()
        self._create_method_requests()
        # ===========================================================================

    def __del__(self):
        if self.check_connection(1, 0.5):
            self.stop_server()

    class RequestTypes(Enum):
        DIRECT="direct"
        CLIENT_STREAMING="client_streaming"
        SERVER_STREAMING="server_streaming"

    def _create_request(self, method_name:str, request_type: RequestTypes):
        CREATE_REQUEST_MAPPING = {SumoRPCClient.RequestTypes.DIRECT: self._request_provider.create_request_method,
                                  SumoRPCClient.RequestTypes.CLIENT_STREAMING: self._request_provider.create_client_streaming_method,
                                  SumoRPCClient.RequestTypes.SERVER_STREAMING: self._request_provider.create_server_streaming_method}
        self._request[method_name] = CREATE_REQUEST_MAPPING[request_type](method_name)

    def _create_helper_requests(self):
        # ---------------------------------------------------------------------------
        # ---------------------------- HELPER REQUESTS ------------------------------

        self._create_request("ping", SumoRPCClient.RequestTypes.DIRECT)
        self._create_request("stop_server", SumoRPCClient.RequestTypes.DIRECT)
        self._create_request("generate_maps_from_cr_file", SumoRPCClient.RequestTypes.DIRECT)

        self._create_request("upload_scenario", SumoRPCClient.RequestTypes.CLIENT_STREAMING)
        self._create_request("download_scenario", SumoRPCClient.RequestTypes.SERVER_STREAMING)
        self._create_request("reset_simulation", SumoRPCClient.RequestTypes.DIRECT)

        # ---------------------------- HELPER REQUESTS ------------------------------
        # ---------------------------------------------------------------------------

    def _create_property_requests(self):
        # ---------------------- SIMULATION PROPERTY REQUESTS -----------------------
        # ---------------------------------------------------------------------------

        sumo_interface_properties = inspect.getmembers(SumoSimulation,
                                                       lambda a: not (inspect.isroutine(a)))
        sumo_interface_public_properties = [tup for tup in sumo_interface_properties if
                                            not tup[0].startswith('_')]
        for sumo_interface_public_variable in sumo_interface_public_properties:
            self._create_request(f"{sumo_interface_public_variable[0]}_get", SumoRPCClient.RequestTypes.DIRECT)
            self._create_request(f"{sumo_interface_public_variable[0]}_set", SumoRPCClient.RequestTypes.DIRECT)

        # ---------------------------------------------------------------------------
        # ---------------------- SIMULATION PROPERTY REQUESTS -----------------------

    def _create_variable_requests(self):
        # ---------------------------------------------------------------------------
        # ---------------------- SIMULATION VARIABLE REQUESTS -----------------------
        n1 = inspect.getmembers(SumoSimulation, lambda a:not(inspect.isroutine(a)))
        n2 = inspect.getmembers(SumoSimulation(), lambda a: not(inspect.isroutine(a)))
        sumo_interface_public_variables = [attr for attr in vars(SumoSimulation()) if
                                           not attr.startswith("_")]
        for sumo_interface_public_variable in sumo_interface_public_variables:
            self._create_request(f"{sumo_interface_public_variable}_get",
                                 SumoRPCClient.RequestTypes.DIRECT)
            self._create_request(f"{sumo_interface_public_variable}_set",
                                 SumoRPCClient.RequestTypes.DIRECT)

        # ---------------------- SIMULATION VARIABLE REQUESTS -----------------------
        # ---------------------------------------------------------------------------

    def _create_method_requests(self):
        # ---------------------------------------------------------------------------
        # ----------------------- SIMULATION METHOD REQUESTS ------------------------

        sumo_interface_methods = inspect.getmembers(SumoSimulation, inspect.isfunction)
        sumo_interface_public_methods = [tup for tup in sumo_interface_methods if
                                         not tup[0].startswith('_')]
        for sumo_interface_public_method in sumo_interface_public_methods:
            self._create_request(sumo_interface_public_method[0],
                                 SumoRPCClient.RequestTypes.DIRECT)

        # ----------------------- SIMULATION METHOD REQUESTS ------------------------
        # ---------------------------------------------------------------------------


    class RequestProvider:
        """Class responsible for providing gRPC requests"""
        _EMPTY_MESSAGE = (None,)

        def __init__(self, channel, module: str, ):
            """Init empty object"""
            self._channel = channel
            self._module = module

        def _grpc_unary_unary_call(self, func, obj_to_bind=None):
            """
            Wrapper function to place the arguments into tuple and get the result from a tuple
            :param func: The function object which need to be wrapped
            :param obj_to_bind: Object to which the function should be bounded
            """

            @wraps(func)
            def func_called_with_arguments(*args):
                if obj_to_bind is not None:
                    func_to_call = partial(func, obj_to_bind)
                else:
                    func_to_call = func

                if len(args) == 0:
                    args = SumoRPCClient.RequestProvider._EMPTY_MESSAGE
                else:
                    args = tuple(args)

                ret_val = func_to_call(args)

                if isinstance(ret_val, tuple) and len(ret_val) == 1 and ret_val[0] is None:
                    ret_val = None
                return ret_val

            return func_called_with_arguments

        def _grpc_stream_unary_call(self, func, obj_to_bind=None):
            """
            Wrapper function to place the arguments into tuple and get the result from a tuple
            :param func: The function object which need to be wrapped
            :param obj_to_bind: Object to which the function should be bounded
            """

            @wraps(func)
            def func_called_with_arguments(*args):
                if obj_to_bind is not None:
                    func_to_call = partial(func, obj_to_bind)
                else:
                    func_to_call = func

                ret_val = func_to_call(*args)
                if isinstance(ret_val, tuple) and len(ret_val) == 1 and ret_val[0] is None:
                    ret_val = None
                return ret_val

            return func_called_with_arguments

        def _grpc_unary_stream_call(self, func, obj_to_bind=None):
            """
            Wrapper function to place the arguments into tuple and get the result from a tuple
            :param func: The function object which need to be wrapped
            :param obj_to_bind: Object to which the function should be bounded
            """

            @wraps(func)
            def func_called_with_arguments(*args):
                if obj_to_bind is not None:
                    func_to_call = partial(func, obj_to_bind)
                else:
                    func_to_call = func

                if len(args) == 0:
                    args = SumoRPCClient.RequestProvider._EMPTY_MESSAGE
                else:
                    args = tuple(args)

                ret_val = func_to_call(args)
                if isinstance(ret_val, tuple) and len(ret_val) == 1 and ret_val[0] is None:
                    ret_val = None
                return ret_val

            return func_called_with_arguments

        @staticmethod
        def _pack(obj) -> bytes:
            """
            Packs the specified object into bytes message
            :param obj: The object to be packed
            """
            try:
                return pickle.dumps(obj)
            except Exception as e:
                SumoRPCClient.logger.critical("Exception occurred during packing the object: {}".format(e))

        @staticmethod
        def _unpack(msg: bytes) -> object:
            """
            Unpacks the specified bytes message into specified object
            :param msg: The message to be unpacked
            """
            try:
                return pickle.loads(msg)
            except Exception as e:
                SumoRPCClient.logger.critical("Exception occurred during unpacking the messsage: {}".format(e))

        def create_request_method(self, method_name):
            """
            Creating RPC request by a method name
            The gRPC methods are unary-unary, therefore the call is wrapped so that the request arguments
            and the reply is placed into tuples.
            :param method_name: The name of the method which call is requested by the gRPC server
            """
            grpc_request = self._channel.unary_unary("/" + self._module + "/" + method_name,
                                                     request_serializer=self._pack,
                                                     response_deserializer=self._unpack)
            return self._grpc_unary_unary_call(grpc_request)

        def create_client_streaming_method(self, method_name):
            """
            Creating RPC request by a method name
            The gRPC methods are unary-unary, therefore the call is wrapped so that the request arguments
            and the reply is placed into tuples.
            :param method_name: The name of the method which call is requested by the gRPC server
            """
            grpc_request = self._channel.stream_unary("/" + self._module + "/" + method_name,
                                                      request_serializer=self._pack,
                                                      response_deserializer=self._unpack)
            return self._grpc_stream_unary_call(grpc_request)

        def create_server_streaming_method(self, method_name):
            """
            Creating RPC request by a method name
            The gRPC methods are unary-unary, therefore the call is wrapped so that the request arguments
            and the reply is placed into tuples.
            :param method_name: The name of the method which call is requested by the gRPC server
            """
            grpc_request = self._channel.unary_stream("/" + self._module + "/" + method_name,
                                                      request_serializer=self._pack,
                                                      response_deserializer=self._unpack)
            return self._grpc_unary_stream_call(grpc_request)

    def stop_server(self) -> bool:
        """Disconnect from the server
        :return: True if the server has been succesfully stopped
        """
        self._request["stop_server"]()
        return self.check_connection(1, 0.5)

    @staticmethod
    def get_trusted_credentials() -> ChannelCredentials:
        """Get SSL credentails for secure communication"""
        # read in certificate
        with open(os.path.join(os.path.dirname(__file__), 'server.crt'), 'rb') as f:
            trusted_certs = f.read()

        # create credentials
        return grpc.ssl_channel_credentials(root_certificates=trusted_certs)

    @Decorators.log_call(level=logging.NOTSET)
    def check_connection(self, num_of_retries: int=3, timeout: float=2.0) -> bool:
        """
        Check the connection to the gRPC server
        :param num_of_retries: The number of retries to connect
        :param timeout: The value of the timeout for a single trial in seconds
        :return: True if the connection is established
        """
        if self.channel is None:
            return False

        count = 0
        while count < num_of_retries:
            try:
                grpc.channel_ready_future(self.channel).result(timeout=timeout)
                self.ping()
                return True
            except (grpc.FutureTimeoutError, grpc.RpcError) as exp:
                count += 1
        return False

    # ============================= GRPC METHODS ================================
    # ---------------------------------------------------------------------------
    # ---------------------------- HELPER METHODS -------------------------------

    @Decorators.log_call(level=logging.NOTSET)
    def ping(self):
        """Check whether the server is alive"""
        self._request["ping"]()

    @Decorators.log_call(level=logging.NOTSET)
    def generate_sumo_scenario(self, scenario_name: str, scenario_path: str) -> AbstractScenarioWrapper:
        """
        Send one commonroad scenario to the server, and request to generate it to the required formats
        :param scenario_name: The name of the scenario to be sent
        :param scenario_path: The absolute path where the scenario can be found
        """
        relative_file_name = f"{scenario_name}/{scenario_name}.cr.xml"
        self._upload(scenario_path, relative_file_name)
        return self._request["generate_maps_from_cr_file"](scenario_name)

    @Decorators.log_call(level=logging.NOTSET)
    def send_sumo_scenario(self, scenario_name: str, scenario_folder: str):
        """
        Send previously generated scenario files to the server
        :param scenario_name: The name of the scenario which will be sent
        :param scenario_folder: The absolute path to the folder containing the files to be sent
        """
        for subdir, dirs, files in os.walk(scenario_folder):
            rel_dir = os.path.relpath(subdir, scenario_folder)
            if rel_dir == '.':
                rel_dir = ''
            for filename in files:
                relative_file_path = os.path.join(scenario_name, rel_dir, filename)
                absolute_file_path = os.path.join(scenario_folder, rel_dir, filename)
                self._upload(absolute_file_path, relative_file_path)

    @Decorators.log_call(level=logging.NOTSET)
    def download_converted_scenarios(self, scenario_name: str, output_directory: str):
        """
        Download generated scneario files from the server
        :param scenario_name: The name of the scenario to be downloaded
        :param output_directory: Absolute path to the directory where the downloaded scenario should be placed
        """
        content_iterator = self._request["download_scenario"](scenario_name)
        output_file_path = os.path.join("/tmp", f"{scenario_name}.zip")
        save_chunks_to_file(content_iterator, output_file_path)
        unzip_file(output_file_path, output_directory)

    @Decorators.log_call(level=logging.NOTSET)
    def _upload(self, source_abs_path: str, dest_rel_path: str):
        """
        Upload a single file to the server's scenario folder
        :param dest_rel_path: Path on the server where the file should be uploaded relative to the server's scenario folder
        :param source_abs_path: Absolute path of the scenario to be sent
        """

        if not os.path.isfile(source_abs_path):
            raise FileNotFoundError(source_abs_path)
        # create a generator
        def upload_data():
            yield dest_rel_path
            yield from get_data_chunks(source_abs_path)

        self._request["upload_scenario"](upload_data())

    @Decorators.log_call(level=logging.NOTSET)
    def reset_simulation(self):
        """
        Resets SumoSimulation.
        """
        self._request["reset_simulation"]()

    # ---------------------------- HELPER METHODS -------------------------------
    # ---------------------------------------------------------------------------
    # ------------------------- SIMULATION PROPERTIES ---------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def current_time_step(self) -> int:
        """ :return: current time step of interface """
        return self._request["current_time_step_get"]()

    @current_time_step.setter
    @Decorators.log_call(level=logging.DEBUG)
    def current_time_step(self, current_time_step):
        """ Time step should not be set manually. """
        self._request["current_time_step_set"](current_time_step)

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def ego_vehicles(self) -> Dict[int, EgoVehicle]:
        """ :return: the ego vehicles of the current simulation. """
        return self._request["ego_vehicles_get"]()

    @ego_vehicles.setter
    @Decorators.log_call(level=logging.DEBUG)
    def ego_vehicles(self, ego_vehicles: Dict[int, EgoVehicle]):
        """ :return: send updated ego vehicles to the current simulation. """
        return self._request["ego_vehicles_set"](ego_vehicles)

    # ------------------------- SIMULATION PROPERTIES ---------------------------
    # ---------------------------------------------------------------------------
    # ---------------------- SIMULATION VARIABLE REQUESTS -----------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def dt(self):
        """ :return: dt of the simulation """
        return self._request["dt"]()

    @dt.setter
    @Decorators.log_call(level=logging.DEBUG)
    def dt(self, dt):
        """ The setter of the dt is not implemented yet """
        raise NotImplementedError('Member variable setter is not supported yet')

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def planning_problem_set(self):
        """ :return: planning_problem_set of the simulation """
        return self._request["planning_problem_set_get"]()

    @planning_problem_set.setter
    @Decorators.log_call(level=logging.DEBUG)
    def planning_problem_set(self, planning_problem_set: PlanningProblemSet):
        """ :return: send updated ego vehicles to the current simulation. """
        self._request["planning_problem_set_set"](planning_problem_set)

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def dummy_ego_simulation(self):
        """ :return: planning_problem_set of the simulation """
        return self._request["dummy_ego_simulation_get"]()

    @dummy_ego_simulation.setter
    @Decorators.log_call(level=logging.DEBUG)
    def dummy_ego_simulation(self, dummy_ego_simulation: bool):
        """ :return: send updated ego vehicles to the current simulation. """
        self._request["dummy_ego_simulation_set"](dummy_ego_simulation)

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def obstacle_states(self):
        """ :return: obstacle_states of the simulation """
        return self._request["obstacle_states_get"]()

    @obstacle_states.setter
    @Decorators.log_call(level=logging.DEBUG)
    def obstacle_states(self, obstacle_states):
        """ The setter of the obstacle_states is not implemented yet """
        self._request["obstacle_states_set"](obstacle_states)

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def simulationdomain(self):
        """ :return: simulationdomain of the simulation """
        return self._request["simulationdomain_get"]()

    @simulationdomain.setter
    @Decorators.log_call(level=logging.DEBUG)
    def simulationdomain(self, simulationdomain):
        """ The setter of the simulationdomain is not implemented yet """
        self._request["simulationdomain_set"](simulationdomain)

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def vehicledomain(self):
        """ :return: vehicledomain of the simulation """
        return self._request["vehicledomain_get"]()

    @vehicledomain.setter
    @Decorators.log_call(level=logging.DEBUG)
    def vehicledomain(self, vehicledomain):
        """ The setter of the vehicledomain is not implemented yet """
        self._request["vehicledomain_set"](vehicledomain)

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def routedomain(self):
        """ :return: routedomain of the simulation """
        return self._request["routedomain_get"]()

    @routedomain.setter
    @Decorators.log_call(level=logging.DEBUG)
    def routedomain(self, routedomain):
        """ The setter of the routedomain is not implemented yet """
        self._request["routedomain_set"](routedomain)

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def ids_sumo2cr(self):
        """ :return: ids_sumo2cr of the simulation """
        return self._request["ids_sumo2cr_get"]()

    @ids_sumo2cr.setter
    @Decorators.log_call(level=logging.DEBUG)
    def ids_sumo2cr(self, ids_sumo2cr):
        self._request["ids_sumo2cr_set"](ids_sumo2cr)

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def ids_cr2sumo(self):
        """ :return: ids_cr2sumo of the simulation """
        return self._request["ids_cr2sumo_get"]()

    @ids_cr2sumo.setter
    @Decorators.log_call(level=logging.DEBUG)
    def ids_cr2sumo(self, ids_cr2sumo):
        self._request["ids_cr2sumo_set"](ids_cr2sumo)

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def obstacle_shapes(self):
        """ :return: obstacle_shapes of the simulation """
        return self._request["obstacle_shapes_get"]()

    @obstacle_shapes.setter
    @Decorators.log_call(level=logging.DEBUG)
    def obstacle_shapes(self, obstacle_shapes):
        self._request["obstacle_shapes_set"](obstacle_shapes)

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def obstacle_types(self):
        """ :return: obstacle_types of the simulation """
        return self._request["obstacle_types_get"]()

    @obstacle_types.setter
    @Decorators.log_call(level=logging.DEBUG)
    def obstacle_types(self, obstacle_types):
        self._request["obstacle_types_set"](obstacle_types)

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def scenarios(self):
        """ :return: example_scenarios of the simulation """
        return self._request["scenarios_get"]()

    @scenarios.setter
    @Decorators.log_call(level=logging.DEBUG)
    def scenarios(self, scenarios):
        self._request["scenarios_set"](scenarios)

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def conf(self):
        """ :return: example_scenarios of the simulation """
        return self._request["conf_get"]()

    @conf.setter
    @Decorators.log_call(level=logging.DEBUG)
    def conf(self, conf):
        return self._request["conf_set"](conf)

    # ---------------------------------------------------------------------------

    @property
    @Decorators.log_call(level=logging.DEBUG)
    def lateral_position_buffer(self):
        """ :return: example_scenarios of the simulation """
        return self._request["lateral_position_buffer_get"]()

    @lateral_position_buffer.setter
    @Decorators.log_call(level=logging.DEBUG)
    def lateral_position_buffer(self, lateral_position_buffer):
        self._request["lateral_position_buffer_set"](lateral_position_buffer)

    # ---------------------- SIMULATION VARIABLE REQUESTS -----------------------
    # ---------------------------------------------------------------------------
    # --------------------------- SIMULATION METHODS ----------------------------

    @Decorators.log_call(level=logging.DEBUG)
    def initialize(self, conf: DefaultConfig,
                   scenario_wrapper: AbstractScenarioWrapper,
                   planning_problem_set: Union[None, PlanningProblemSet] = None) -> None:
        """
        Reads scenario files, starts traci simulation, initializes vehicles, conducts pre-simulation.

        :param conf: configuration object. If None, use default configuration.
        :param scenario_wrapper: handles all files required for simulation.
        :param start_logging: if logging was initialized before, set Falseconf
        """
        base_dir = os.path.join("scenarios", conf.scenario_name)
        scenario_wrapper.sumo_cfg_file = os.path.join(base_dir,
                                          os.path.split(scenario_wrapper.sumo_cfg_file)[1])
        self._request["initialize"](conf, scenario_wrapper, planning_problem_set)

    @Decorators.log_call(level=logging.DEBUG)
    def init_ego_vehicles_from_planning_problem(self, planning_problem_set: PlanningProblemSet) -> None:
        """
        Initializes the ego vehicles according to planning problem set.

        :param planning_problem_set: The planning problem set which defines the ego vehicles.
        """
        self._request["init_ego_vehicles_from_planning_problem"](planning_problem_set)

    @Decorators.log_call(level=logging.DEBUG)
    def print_lanelet_net(self, with_lane_id=True, with_succ_pred=False, with_adj=False, with_speed=False):
        """
        Plots commonroad road network without vehicles or obstacles.

        :param with_lane_id: if set to true, the lane id will also be printed.
        :param with_succ_pred: if set to true, the successor and predecessor lanelets will be printed.
        :param with_adj: if set to true, adjacant lanelets will be printed.
        :param with_speed: if set to true, the speed limit will be printed.
        """
        raise NotImplementedError("Plot from Docker is not implemented yet")

    @Decorators.log_call(level=logging.DEBUG)
    def commonroad_scenario_at_time_step(self, time_step: int, add_ego=False, start_0=True) -> Scenario:
        """
        Creates and returns a commonroad scenario at the given time_step. Initial time_step=0 for all obstacles.
        :param time_step: the scenario will be created according this time step.
        :param add_ego: whether to add ego vehicles to the scenario.
        :param start_0: if set to true, initial time step of vehicles is 0, otherwise, the current time step
        """
        return self._request["commonroad_scenario_at_time_step"](time_step, add_ego, start_0)

    @Decorators.log_call(level=logging.DEBUG)
    def commonroad_scenarios_all_time_steps(self, lanelet_network: LaneletNetwork = None) -> Scenario:
        """
        Creates and returns a commonroad scenario with all the dynamic obstacles.
        :param lanelet_network:
        :return: list of cr example_scenarios, list of cr example_scenarios with ego, list of planning problem sets)
        """
        return self._request["commonroad_scenarios_all_time_steps"](lanelet_network)

    @Decorators.log_call(level=logging.DEBUG)
    def simulate_step(self) -> None:
        """ Executes next simulation step (consisting of delta_steps sub-steps with dt_sumo=dt/delta_steps) in SUMO """
        self._request["simulate_step"]()

    @Decorators.log_call(level=logging.DEBUG)
    def get_ego_obstacles(self, time_step: Union[int, None] = None) -> List[DynamicObstacle]:
        """
        Get list of ego vehicles converted to Dynamic obstacles
        :param time_step: initial time step, if None, get complete driven trajectory
        :return:
        """
        return self._request["get_ego_obstacles"](time_step)

    @Decorators.log_call(level=logging.DEBUG)
    def stop(self) -> None:
        """ Exits SUMO Simulation """
        self._request["stop"]()

    @Decorators.log_call(level=logging.DEBUG)
    def check_lanelets_future_change(self, current_state: State, planned_traj: List[State]) -> Tuple[str, int]:
        """
        Checks the lanelet changes of the ego vehicle in the future time_window.

        :param lanelet_network: object of the lanelet network
        :param time_window: the time of the window to check the lanelet change
        :param traj_index: index of the planner output corresponding to the current time step

        :return: lc_status, lc_duration: lc_status is the status of the lanelet change in the next time_window; lc_duration is the unit of time steps (using sumo dt)

        """
        return self._request["check_lanelets_future_change"](current_state, planned_traj)

    # --------------------------- SIMULATION METHODS ----------------------------
    # ---------------------------------------------------------------------------
    # ===========================================================================
