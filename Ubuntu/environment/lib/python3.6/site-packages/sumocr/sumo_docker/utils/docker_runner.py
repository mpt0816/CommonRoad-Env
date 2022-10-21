import os
import logging
import time
from threading import Thread
from typing import List
from packaging import version
import docker
from collections import defaultdict
from docker import APIClient
import docker.errors as errors
from docker.models.containers import Container
from docker.types.daemon import CancellableStream
from docker.models.images import Image
import getpass
from  enum import Enum

__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["ZIM Projekt ZF4086007BZ8"]
__version__ = "0.6.0"
__maintainer__ = "Peter Kocsis"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Testing"


class DockerRunner:
    """ Class for managing Docker container"""
    DOCKER_TAG_DELIMITER = ':'
    DOCKER_REPO_DELIMITER = '/'
    DOCKER_CONTAINER_NAME_DELIMITER = '-'
    DOCKER_IMAGE_TAG_PREFIX = 'ver'

    logger = logging.getLogger('crsumo.docker_runner')
    logger.setLevel(logging.DEBUG)

    class Decorators:
        """ Class for managing decorators used """

        @classmethod
        def handle_docker_exception(cls, func):
            """
            Exception handler decorator
            :param func: The function which exceptions should be handled
            """

            def func_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except errors.NotFound as e:
                    DockerRunner.logger.warning(e.explanation)
                    pass
                except errors.APIError as e:
                    DockerRunner.logger.critical(
                        "Docker APIError occured, please ensure that you can run the docker without root permissions")
                    raise e
                return None

            return func_wrapper

    class DockerLogHandler(Thread):
        """ Class for managing the logs of a Docker container """

        def __init__(self, log_generator: CancellableStream):
            """
            Init empty object
            :param log_generator: Stream of the logs which will be handled
            """
            super().__init__()
            self.log_generator = log_generator

        def __del__(self):
            """ Destruct the object """
            self.stop()

        def run(self):
            """ Run the log handling """
            DockerRunner.logger.debug("Starting Docker logger")
            for log in self.log_generator:
                DockerRunner.logger.info("DockerMsg: {}".format(log.decode("utf-8")).rstrip())
            DockerRunner.logger.debug("Docker logger ended")

        def stop(self):
            """ Stop the log handling """
            DockerRunner.logger.debug("Stopping Docker logger")
            self.log_generator.close()
            DockerRunner.logger.debug("Docker logger stopped")

    class DockerStatus(Enum):
        """ Status of a Docker container """
        CREATED = 'created'
        RESTARTING = 'restarting'
        RUNNING = 'running'
        REMOVING = 'removing'
        PAUSED = 'paused'
        EXITED = 'exited'
        DEAD = 'dead'

    def __init__(self):
        """Init empty"""
        logging.basicConfig()
        self.client = self._get_docker_client()
        self._check_api_connection()
        self.version = self.get_version()

        self.image_name = None
        self.image_tag = None
        self.image_full_name = None
        self.image = None
        self.server_log_level = None
        self.docker_registry_address = None
        self.container = None
        self.docker_logger: DockerRunner.DockerLogHandler or None = None

    def prepare_docker(self, image_name: str, image_version: str,
                       server_log_level='info', docker_registry_address: str=None):
        """
        Prepares to start container with image
        The desired image tag will be DOCKER_IMAGE_TAG_PREFIX + image_version
        :param image_name: The name of the used image
        :param image_version: The version of the image, if None is passed, the latest local version will be used
        :param server_log_level: The log level of the server, set to 'debug' for detailed logs
        """
        self.image_name = image_name
        self.server_log_level = server_log_level

        self.docker_registry_address = docker_registry_address
        if self.docker_registry_address is not None:
            if not self.check_login():
                self.login()

        if image_version is None:
            self.image = self.get_image(self.image_name)
            self.image_tag = self.get_latest_tag(self.image)
            self.image_full_name = self.image_name + self.DOCKER_TAG_DELIMITER + self.image_tag
        else:
            self.image_tag = image_version
            self.image_full_name = self.image_name + self.DOCKER_TAG_DELIMITER + self.image_tag
            self.image = self.get_image(self.image_full_name)

    def _get_docker_client(self):
        try:
            return docker.from_env()
        except errors.DockerException as e:
            self.logger.critical("Cannot connect to Docker! "
                                 "Ensure that docker is installed, running and it doesn't require root privileges!")
            raise e

    def _check_api_connection(self):
        if self.client is None or self.client.ping() is False:
            raise RuntimeError("Docker runtime is not responding!")

    @Decorators.handle_docker_exception
    def get_version(self):
        """ Get the version of the docker client """
        return self.client.version()

    @Decorators.handle_docker_exception
    def login(self):
        """
        Log in to the docker registry
        :param docker_registry_address: The docker registry to log in
        """
        self.logger.info(f"You haven't logged in to {self.docker_registry_address} yet, log in now")
        username = input(f"Username to {self.docker_registry_address}: ")
        password = getpass.getpass(prompt='Password: ')

        # Docker Client login doesn't update the config file, so everytime a login will be required
        # So using shell command instead of the Docker API call
        os.system(f"docker login {self.docker_registry_address} --username {username} --password {password}")
        self.client = docker.DockerClient() # refresh the client

    @Decorators.handle_docker_exception
    def check_login(self) -> bool:
        """
        Checks whether already logged in to the docker registry or not
        :param docker_registry_address: The docker registry to log in
        """
        docker_config = os.path.expanduser('~/.docker/config.json')
        if os.path.exists(docker_config):
            with open(docker_config) as f:
                return self.docker_registry_address in f.read()
        else:
            return False

    @staticmethod
    def get_latest_tag(image: Image) -> str:
        """
        Gets the latest version of the image which is available locally.
        If no image with specified version is found,
        it will uses the 'latest' tag (which means not always the latest version)
        :param image: The image which latest version is to be found
        :return: The tag of the latest version
        """
        image_name_delimiter = DockerRunner.DOCKER_TAG_DELIMITER + DockerRunner.DOCKER_IMAGE_TAG_PREFIX
        tags = image.tags
        splited_tags = {tuple(tag.split(image_name_delimiter)) for tag in tags}
        versions = {tag_value[1] if len(tag_value) == 2 else None for tag_value in splited_tags}
        if all(v is None for v in versions):

            return 'latest'
        else:
            latest_version = max(versions, key=lambda x: version.parse(x))
            return DockerRunner.DOCKER_IMAGE_TAG_PREFIX + latest_version

    def get_image(self, image_full_name: str) -> Image:
        """
        Gets the specified image. If it is not found locally, it will be pulled
        :param image_full_name: The full name of the image to be queried
        :return: The found local Docker image
        """
        image = self.query_image(image_full_name)
        if image is None:
            self.logger.info("Image {} not found, pulling it".format(image_full_name))
            image = self.pull_image(image_full_name)
            self.logger.info("Image {} pulled".format(image_full_name))
            image = self.query_image(image_full_name)

        self.logger.info("Image {} found".format(image_full_name))
        return image

    @Decorators.handle_docker_exception
    def query_image(self, image_full_name: str) -> Image or None:
        """
        Queries the specified image locally
        :param image_full_name: The full name of the image to be queried
        :return: The found local Docker image if it is found locally, None otherwise
        """
        image_list = self.client.images.list(name=image_full_name)
        if len(image_list) == 0:
            return None
        else:
            return image_list[0]

    def pull_image(self, image_full_name: str):
        """
        Pulls image from DockerHub
        :param image_full_name: The full name of the image to be queried
        :return: The pulled Docker image
        """
        # Docker Client pull doesn't provide logs. To receive the logs, the low-level API should be used,
        # but it just prints the states without any structure. Therefore call the shell command directly
        # to receive the logs
        os.system(f"docker pull {image_full_name}")
        # if image is None:
        #     raise errors.NotFound("Can't pull image {}, it was not found remotely".format(image_full_name))
        # return image

    def get_fresh_container(self, container_name: str) -> Container:
        """
        Preparing a new container which will be used
        :param container_name: The desired name of the container
        :return: The created Docker container
        """
        container = self.query_container(container_name)
        if container is None:
            self.logger.info("Container {} not found, creating it".format(container_name))
        else:
            self.logger.info("Container {} found, removing it".format(container_name))
            status = self.get_container_status()
            if self.DockerStatus.CREATED == status:
                self.logger.debug('Unused docker container found, removing it')
                self.remove_container(container)
                self.logger.debug("Container {} stopped and removed".format(container_name))
            elif self.DockerStatus.RESTARTING == status:
                self.logger.debug('Restarting docker container found, stopping it and removing it')
                self.stop_container(container)
                self.remove_container(container)
                self.logger.debug("Container {} stopped and removed".format(container_name))
            elif self.DockerStatus.RUNNING == status:
                self.logger.debug('Running docker container found, stopping and removing it')
                self.stop_container(container)
                self.remove_container(container)
                self.logger.debug("Container {} stopped and removed".format(container_name))
            elif self.DockerStatus.REMOVING == status:
                self.logger.debug('Removing docker container found')
            elif self.DockerStatus.PAUSED == status:
                self.logger.debug('Paused docker container found, stopping and removing it')
                self.stop_container(container)
                self.remove_container(container)
                self.logger.debug("Container {} stopped and removed".format(container_name))
            elif self.DockerStatus.EXITED == status:
                self.logger.debug('Exited docker container found, removing it')
                self.remove_container(container)
            elif self.DockerStatus.DEAD == status:
                self.logger.debug('Paused docker container found, removing it')
                self.remove_container(container)
        container = self.create_container(container_name)
        self.logger.info("Container {} created".format(container_name))
        return container

    @Decorators.handle_docker_exception
    def query_container(self, container_name: str) -> Container or None:
        """
        Queries a container locally
        :param container_name: The name of the searched container
        :return: Docker container if it was found, None otherwise
        """
        return self.client.containers.get(container_name)

    @Decorators.handle_docker_exception
    def create_container(self, container_name: str = None) -> Container:
        """
        Creating new container with the desired name
        :param container_name: The desired name of the container
        :return: The created Docker container
        """
        return self.client.containers.create(self.image_full_name, name=container_name, auto_remove=True,
                                             publish_all_ports=True,
                                             environment=[f"rpc_server_log_level={self.server_log_level}"])

    @Decorators.handle_docker_exception
    def remove_container(self, container: Container):
        """
        Removing the container
        :param container: The container to be deleted
        """
        container.remove()

    def get_container_status(self) -> DockerStatus:
        """
        Returns the status of the container
        """
        self.container.reload()
        return self.DockerStatus(self.container.status)

    @Decorators.handle_docker_exception
    def start(self):
        """ Start the container and subscribe for the logs """
        self.container = self.create_container()
        self.container.start()
        self.container.reload()
        self.docker_logger = DockerRunner.DockerLogHandler(self.container.logs(stream=True, timestamps=True))
        self.docker_logger.start()
        self.logger.info("Container started with ID: {}".format(self.container.id))

    @Decorators.handle_docker_exception
    def stop(self):
        """ Stop the container """
        self.stop_container(self.container)
        self.docker_logger.stop()
        self.logger.info("Container {} stopped".format(self.container.id))

    @Decorators.handle_docker_exception
    def stop_container(self, container: Container):
        """
        Stopping a container
        :param container: The container to be stopped
        """
        container.stop(timeout=2)
        container.wait()
        container.reload()

    @Decorators.handle_docker_exception
    def get_ip(self) -> str:
        """
        Get the IP of the running container
        """
        ip_address = self.container.attrs["NetworkSettings"]["IPAddress"]
        return ip_address

    @Decorators.handle_docker_exception
    def get_host_address(self) -> dict:
        """
        Get the published Host IP of the running container
        """
        ip_addresses = self.container.ports
        return ip_addresses

    @Decorators.handle_docker_exception
    def get_exposed_ports(self) -> List[str]:
        """
        Get the exposed ports of the running container
        """
        exposed_ports = self.container.ports.keys()
        exposed_ports = {port.split('/')[0] for port in exposed_ports}
        return list(exposed_ports)
