from sumocr import DOCKER_REGISTRY
from sumocr.sumo_docker.rpc.sumo_client import SumoRPCClient
from sumocr.sumo_docker.utils.docker_runner import DockerRunner

__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["ZIM Projekt ZF4086007BZ8"]
__version__ = "0.7.0"
__maintainer__ = "Peter Kocsis"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Testing"


class SumoInterface:
    """Class for setting up a simulation with a dockerized Sumo installation."""

    def __init__(self, use_docker: bool = True):
        """Init empty object"""
        self.use_docker = use_docker
        self.sumo_client: SumoRPCClient or None = None
        if self.use_docker:
            self._sumo_container = DockerRunner()
        else:
            self._sumo_container = None

    def start_simulator(self) -> SumoRPCClient:
        """
        Starting SUMO in docker container and return a client for connecting with it.
        :return: SumoRPCClient which can communicate with SUMO
        """
        if self.use_docker:
            self._sumo_container.prepare_docker(DOCKER_REGISTRY,
                                                "ver{}".format(SumoRPCClient.__version__),
                                                server_log_level='info',
                                                docker_registry_address='gitlab.lrz.de:5005')
            self._sumo_container.start()
            container_address_dict = self._sumo_container.get_host_address()
            container_address_value = container_address_dict['50051/tcp'][0]
            container_address = f"localhost:{container_address_value['HostPort']}"
        else:
            container_address = "localhost:50051"

        self.sumo_client = SumoRPCClient(container_address)
        self.sumo_client.ping()
        return self.sumo_client

    def stop_simulator(self) -> None:
        """
        Stop the sumo simulation and stop the docker container.
        :return: None
        """
        self.sumo_client.stop_server()
        if self.sumo_client is not None:
            self.sumo_client = None

        if self._sumo_container is not None:
            self._sumo_container.stop()

    def check_docker_health(self) -> bool:
        """
        Checks the health of the docker container
        :return: True if the container is up and running
        """
        status = self._sumo_container.get_container_status()
        return status == DockerRunner.DockerStatus.RUNNING
