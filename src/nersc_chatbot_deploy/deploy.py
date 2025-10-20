"""
Deployment utilities for managing LLM serving jobs on NERSC Slurm clusters.

This module provides functions to allocate GPU resources, deploy LLM models using
specified backends (currently vLLM), monitor job and service status, generate API keys,
and check service health via RESTful APIs.
"""

import json
import logging
import os
import queue
import secrets
import shlex
import string
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from math import ceil
from typing import IO, Dict, Optional, Tuple

import requests
from rich.box import ROUNDED
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)


class ProcessLogger:
    """
    Manages background threads to capture stdout and stderr from a subprocess,
    writing output to both a log file and the console in real-time.
    """

    def __init__(
        self,
        process: subprocess.Popen,
        log_file_path: Optional[str],
        job_name: str,
        use_rich_display: bool = False,
        max_display_lines: int = 20,
    ):
        """
        Initializes the ProcessLogger and starts daemon reader threads.

        Args:
            process (subprocess.Popen): The subprocess to monitor.
            log_file_path (Optional[str]): Path to the log file.
            job_name (str): Name of the job (for logging context).
            use_rich_display (bool): Whether to use Rich Live display for console output.
            max_display_lines (int): Maximum number of lines to display in Rich panel.
        """
        self.process = process
        self.log_file_path = log_file_path
        self.job_name = job_name
        self.stdout_queue: queue.Queue[Tuple[str, Optional[str]]] = queue.Queue()
        self.stderr_queue: queue.Queue[Tuple[str, Optional[str]]] = queue.Queue()
        self.log_file = None
        self.stdout_thread = None
        self.stderr_thread = None
        self.use_rich_display = use_rich_display
        self.max_display_lines = max_display_lines
        self.live = None

        # Open the log file
        if self.log_file_path:
            try:
                self.log_file = open(self.log_file_path, "w", encoding="utf-8")
                logger.info(
                    f"Logging output for job '{job_name}' to: {self.log_file_path}"
                )
            except Exception as e:
                logger.error(f"Failed to open log file {self.log_file_path}: {e}")

        # Initialize Rich display if enabled
        if self.use_rich_display:
            self.display_deque = deque(
                [""] * max_display_lines, maxlen=max_display_lines
            )
            self.console = Console()
            self.live = Live(console=self.console, screen=False, auto_refresh=False)
            try:
                self.live.start()
                logger.debug(f"Started Rich Live display for job '{job_name}'")
            except Exception as e:
                logger.error(f"Failed to start Rich display: {e}")
                self.use_rich_display = False  # Fallback to regular print

        # Start reader threads as daemons
        self.stdout_thread = threading.Thread(
            target=self._reader_thread,
            args=(self.process.stdout, self.stdout_queue, "STDOUT"),
            daemon=True,
        )
        self.stderr_thread = threading.Thread(
            target=self._reader_thread,
            args=(self.process.stderr, self.stderr_queue, "STDERR"),
            daemon=True,
        )

        self.stdout_thread.start()
        self.stderr_thread.start()

        # Start a processing thread to handle output
        self.processor_thread = threading.Thread(
            target=self._process_output,
            daemon=True,
        )
        self.processor_thread.start()

    def _reader_thread(
        self, pipe: IO[str], q: queue.Queue[Tuple[str, Optional[str]]], stream_name: str
    ) -> None:
        """
        Reads lines from a pipe and puts them into a queue.

        Args:
            pipe: The pipe to read from (stdout or stderr).
            q (queue.Queue): The queue to put lines into.
            stream_name (str): Name of the stream for logging.
        """
        try:
            with pipe:
                for line in iter(pipe.readline, ""):
                    if line:
                        q.put((stream_name, line))
        except Exception as e:
            logger.error(f"Error reading from {stream_name}: {e}")
        finally:
            q.put((stream_name, None))  # Signal that the pipe has closed

    def _process_output(self) -> None:
        """
        Processes output from both stdout and stderr queues and writes to file and console.
        """
        stdout_closed = False
        stderr_closed = False

        while not stdout_closed or not stderr_closed:
            # Process all available stdout lines
            while not stdout_closed:
                try:
                    stream_name, line = self.stdout_queue.get(timeout=0.001)
                    if line is None:
                        stdout_closed = True
                    else:
                        self._write_output(stream_name, line)
                except queue.Empty:
                    break  # No more stdout data available right now

            # Process all available stderr lines
            while not stderr_closed:
                try:
                    stream_name, line = self.stderr_queue.get(timeout=0.001)
                    if line is None:
                        stderr_closed = True
                    else:
                        self._write_output(stream_name, line)
                except queue.Empty:
                    break  # No more stderr data available right now

            # Small sleep only when both queues are empty
            if not stdout_closed or not stderr_closed:
                time.sleep(0.01)

        logger.debug(f"Output processing completed for job '{self.job_name}'")

    def _write_output(self, stream_name: str, line: str) -> None:
        """
        Writes a line of output to the log file and console.

        Args:
            stream_name (str): Name of the stream (STDOUT or STDERR).
            line (str): The line to write.
        """
        # File logging
        if self.log_file:
            self.log_file.write(line)
            self.log_file.flush()

        # Console output
        if self.use_rich_display:
            self._update_rich_display(stream_name, line)
        else:
            if stream_name == "STDERR":
                print(f"{line}", end="", file=sys.stderr)
            else:
                print(line, end="")
            sys.stdout.flush()

    def _update_rich_display(self, stream_name: str, line: str) -> None:
        """
        Updates the Rich Live display with a new line of output.

        Args:
            stream_name (str): Name of the stream (STDOUT or STDERR).
            line (str): The line to display.
        """
        try:
            # Format line with optional STDERR prefix
            if stream_name == "STDERR":
                display_line = f"[red]STDERR:[/red] {line.rstrip()}"
            else:
                display_line = line.rstrip()

            # Add to deque (oldest line automatically removed if full)
            self.display_deque.append(display_line)

            # Create renderable from deque
            log_text = Text.from_markup("\n".join(self.display_deque))

            panel = Panel(
                log_text,
                title=f"🚀 {self.job_name}",
                border_style="blue",
                box=ROUNDED,
                padding=(0, 1),
            )

            # Update live display
            if self.live:
                self.live.update(panel, refresh=True)

        except Exception as e:
            logger.error(f"Error updating Rich display: {e}")
            # Fallback to stderr for this line
            print(f"STDERR: {line}" if stream_name == "STDERR" else line, end="")

    def cleanup(self) -> None:
        """
        Waits for the process and threads to finish, then closes the log file and stops Rich display.
        """
        logger.info(f"Cleaning up logger for job '{self.job_name}'")

        # Wait for threads to finish
        if self.stdout_thread and self.stdout_thread.is_alive():
            self.stdout_thread.join(timeout=5)
        if self.stderr_thread and self.stderr_thread.is_alive():
            self.stderr_thread.join(timeout=5)
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)

        # Stop Rich display
        if self.use_rich_display and self.live:
            try:
                self.live.stop()
                logger.debug(f"Stopped Rich Live display for job '{self.job_name}'")
            except Exception as e:
                logger.error(f"Error stopping Rich display: {e}")

        # Close the log file
        if self.log_file:
            self.log_file.close()
            logger.info(f"Log file closed: {self.log_file_path}")


def allocate_gpu_resources(
    account: str,
    num_gpus: int,
    queue: str,
    time: str,
    job_name: str,
    commands: str,
    constraint: str = "gpu",
    num_nodes: int = 1,
    log_output: bool = True,
    use_rich_display: bool = False,
    max_display_lines: int = 20,
) -> Tuple[Optional[subprocess.Popen], Optional[ProcessLogger]]:
    """
    Executes the command to allocate GPU resources using the `salloc` command, and runs commands within the allocated session in the background.

    The command executed is:
    `salloc -N <num_nodes> -C <constraint> -G <num_gpus> -t <time> -q <queue> -A <account> -J <job_name> /bin/bash -c '<commands>'`

    Args:
        account (str): The account name to be used with the `-A` option.
        num_gpus (int): The number of GPUs to request with the `-G` option.
        queue (str): The queue to use with the `-q` option.
        time (str): The time to allocate with the `-t` option (formatted as HH:MM:SS).
        job_name (str): The job name to be used with the `-J` option.
        commands (str): Additional commands to run within the allocated session.
        constraint (str, optional): The constraint to use with the `-C` option. Defaults to "gpu".
        num_nodes (int, optional): The number of nodes to request with the `-N` option. Defaults to 1.
        log_output (bool, optional): Whether to log the output to a file. Defaults to True.
        use_rich_display (bool, optional): Whether to use Rich Live display for console output. Defaults to False.
        max_display_lines (int, optional): Maximum number of lines to display in Rich panel. Defaults to 20.

    Returns:
        Tuple[Optional[subprocess.Popen], Optional[ProcessLogger]]: The Popen object representing the background process and the ProcessLogger object, or (None, None) if an error occurs.

    Note:
        The subprocess.Popen runs asynchronously; the caller is responsible for managing the process lifecycle.
    """
    # Quote the account name, queue, time, job name, and additional commands to handle any special characters or spaces
    quoted_account = shlex.quote(account)
    quoted_queue = shlex.quote(queue)
    quoted_time = shlex.quote(time)
    quoted_job_name = shlex.quote(job_name)
    quoted_commands = shlex.quote(commands)
    quoted_constraint = shlex.quote(constraint)

    # Define the command string with the quoted arguments
    command_str = f"salloc -N {num_nodes} -C {quoted_constraint} -G {num_gpus} -t {quoted_time} -q {quoted_queue} -A {quoted_account} -J {quoted_job_name} /bin/bash -c {quoted_commands}"

    # Split the command string into a list of arguments using shlex.split
    command = shlex.split(command_str)
    logger.debug(f"Allocating GPU resources with command: {' '.join(command)}")

    try:
        # Execute the command using subprocess.Popen to run it in the background
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        logger.info(f"Started process with PID: {process.pid}")

        # Create ProcessLogger if logging is enabled
        process_logger = None
        log_file_path = None
        if log_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = f"{job_name}_{timestamp}.log"

        process_logger = ProcessLogger(
            process,
            log_file_path,
            job_name,
            use_rich_display=use_rich_display,
            max_display_lines=max_display_lines,
        )

        return process, process_logger
    except Exception as e:
        # Handle any exceptions
        logger.error(f"An error occurred while starting the command: {e}")
        return None, None


def generate_api_key(length: int = 32) -> str:
    """
    Generates a random API key string.

    Args:
        length (int): The length of the API key. Default is 32 characters.

    Returns:
        str: The generated API key.
    """
    logger.debug(f"Generating API key of length {length}")

    # Define the characters to use in the API key
    characters = string.ascii_letters + string.digits

    # Generate the API key
    api_key = "".join(secrets.choice(characters) for _ in range(length))

    return api_key


def deploy_llm(
    account: str,
    num_gpus: int,
    queue: str,
    time: str,
    job_name: str,
    model: str,
    backend: str = "vllm",
    backend_args: Dict[str, str] = {},
    constraint: str = "gpu",
    log_output: bool = True,
    use_rich_display: bool = False,
    max_display_lines: int = 20,
) -> Tuple[Optional[subprocess.Popen], str, Optional[ProcessLogger]]:
    """
    Deploys the LLM using the specified backend with the provided parameters.

    Args:
        account (str): The account name to be used with the `-A` option.
        num_gpus (int): The number of GPUs to request with the `-G` option.
        queue (str): The queue to use with the `-q` option.
        time (str): The time to allocate with the `-t` option (formatted as HH:MM:SS).
        job_name (str): The job name to be used with the `-J` option.
        model (str): The model to be served by the backend.
        backend (str, optional): The backend to use (defaults to "vllm").
        backend_args (Dict[str, str], optional): Additional arguments to pass to the backend.
        constraint (str, optional): The constraint to use with the `-C` option. Defaults to "gpu".
        log_output (bool, optional): Whether to log the output to a file. Defaults to True.
        use_rich_display (bool, optional): Whether to use Rich Live display for console output. Defaults to False.
        max_display_lines (int, optional): Maximum number of lines to display in Rich panel. Defaults to 20.

    Returns:
        Tuple[Optional[subprocess.Popen], str, Optional[ProcessLogger]]: The Popen object representing the background process, the API key (if applicable), and the ProcessLogger object, or (None, "", None) if an error occurs.

    Raises:
        ValueError: If multi-node deployment is requested (unsupported).
    """
    logger.info(
        f"Deploying LLM model '{model}' on account '{account}' with {num_gpus} GPUs, job name '{job_name}'"
    )
    logger.debug(f"Backend args: {backend_args}")

    llm_api_key = ""

    # Calculate the number of nodes required
    num_nodes = ceil(num_gpus / 4)
    logger.debug(f"Calculated num_nodes: {num_nodes}")

    # Raise an error if multi-node is requested (unsupported)
    if num_nodes > 1:
        logger.error("Multi-node deployment is currently unsupported")
        raise ValueError("Multi-node deployment is currently unsupported")

    # Calculate the number of CPUs per task
    cpus_per_task = min(num_gpus * 32, 128)

    # Calculate the number of GPUs per task
    gpus_per_task = min(num_gpus, 4)
    logger.debug(f"CPUs per task: {cpus_per_task}, GPUs per task: {gpus_per_task}")

    # Get the HF_TOKEN and HF_HOME from the environment for Hugging Face authentication
    hf_token = os.getenv("HF_TOKEN")
    hf_home = os.getenv("HF_HOME")
    logger.debug(
        f"HF_TOKEN set: {'Yes' if hf_token else 'No'}, HF_HOME set: {'Yes' if hf_home else 'No'}"
    )

    # Construct the backend command with the provided arguments
    if backend == "vllm":
        # Generate an API key for the LLM service
        llm_api_key = generate_api_key()
        logger.info("Generated API key for LLM service")

        # Get the vLLM_IMAGE from the environment or use default
        vllm_image = os.getenv("vLLM_IMAGE", "vllm/vllm-openai:v0.11.0")
        logger.debug(f"Using vLLM image: {vllm_image}")

        backend_command = (
            f"srun -n 1 --cpus-per-task={cpus_per_task} --gpus-per-task={gpus_per_task} "
            "  shifter "
            f"    --image={vllm_image} "
            "    --module=gpu,nccl-plugin "
            f"{f'--env=HF_TOKEN={hf_token}' if hf_token else ''} "
            f"{f'--env=HF_HOME={hf_home}' if hf_home else ''} "
            f"        vllm serve {model} "
            f"             --api-key {llm_api_key}"
        )
        logger.debug(f"Backend command before adding args: {backend_command}")
    else:
        logger.error(f"Unsupported backend: {backend}")
        raise ValueError(f"Unsupported backend: {backend}")

    # Add additional backend arguments
    for arg, value in backend_args.items():
        backend_command += f" --{arg} {value}"
    logger.debug(f"Final backend command: {backend_command}")

    # Allocate GPU resources via Slurm and start the LLM process
    process, process_logger = allocate_gpu_resources(
        account=account,
        num_gpus=num_gpus,
        queue=queue,
        time=time,
        job_name=job_name,
        commands=backend_command,
        constraint=constraint,
        num_nodes=num_nodes,
        log_output=log_output,
        use_rich_display=use_rich_display,
        max_display_lines=max_display_lines,
    )
    logger.info(f"Started Slurm job with PID: {process.pid if process else 'None'}")

    return process, llm_api_key, process_logger


def get_node_address(job_name: str) -> Optional[str]:
    """
    Gets the node address from a named Slurm job.

    Args:
        job_name (str): The name of the Slurm job.

    Returns:
        Optional[str]: The node address if found, None otherwise.
    """
    logger.info(f"Fetching node address for job: {job_name}")
    try:
        # Get the job ID of the named job
        squeue_cmd = (
            f"squeue --name={shlex.quote(job_name)} --me --state=RUNNING -h -o %A"
        )
        job_id = subprocess.check_output(shlex.split(squeue_cmd), text=True).strip()

        if not job_id:
            logger.info(f"Not yet found running job with the name {job_name}")
            return None

        # Get the node address from the job ID
        scontrol_cmd = f"scontrol show job {job_id} --json"
        scontrol_output = subprocess.check_output(shlex.split(scontrol_cmd), text=True)
        job_info = json.loads(scontrol_output)
        node_address = job_info["jobs"][0]["nodes"]

        return node_address
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred while executing the command: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"An error occurred while decoding JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None


def check_service_status(
    api_url: str,
    endpoint: str = "/models",
    api_key: Optional[str] = None,
    expected_status: int = 200,
) -> bool:
    """
    Checks if a service has started up via its RESTful API.

    Args:
        api_url (str): The base URL of the service's RESTful API.
        endpoint (str): The endpoint to check the status. Default is "/models".
        api_key (Optional[str]): The API key to use for the request. Default is None.
        expected_status (int): The expected HTTP status code. Default is 200.

    Returns:
        bool: True if the service is up, False otherwise.
    """
    logger.debug(f"Checking service status at {api_url}{endpoint}")

    # Construct the full URL for the endpoint
    url = f"{api_url.rstrip('/')}{endpoint}"

    # Set up headers with the API key if provided
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Make the GET request to the endpoint
    response = requests.get(url, headers=headers)

    # Check if the response status code matches the expected status
    if response.status_code == expected_status:
        return True
    else:
        logger.warning(f"Received unexpected status code: {response.status_code}")
        return False


def monitor_job_and_service(
    job_name: str,
    process: Optional[subprocess.Popen] = None,
    api_url_template: str = "http://{node_address}:8000/v1",
    endpoint: str = "/models",
    api_key: Optional[str] = None,
    expected_status: int = 200,
    job_timeout: int = 300,
    service_timeout: int = 300,
    job_interval: int = 30,
    service_interval: int = 30,
) -> Optional[str]:
    """
    Monitors the Slurm job and service status by checking every interval seconds and times out after timeout seconds.

    Args:
        job_name (str): The name of the Slurm job.
        process (Optional[subprocess.Popen]): The subprocess to monitor. If provided,
            monitoring will fail if the process exits. Defaults to None.
        api_url_template (str): The template for the API URL with a `{node_address}` placeholder.
            Defaults to "http://{node_address}:8000/v1".
        endpoint (str): The endpoint to check the status. Default is "/models".
        api_key (Optional[str]): The API key to use for the request. Default is None.
        expected_status (int): The expected HTTP status code. Default is 200.
        job_timeout (int): The maximum time to wait for the job in seconds. Default is 300 seconds (5 minutes).
        service_timeout (int): The maximum time to wait for the service in seconds. Default is 300 seconds (5 minutes).
        job_interval (int): The interval between job checks in seconds. Default is 30 seconds.
        service_interval (int): The interval between service checks in seconds. Default is 30 seconds.

    Returns:
        Optional[str]: The LLM address if both the job and service are running, None otherwise.

    Note:
        This function polls the Slurm job status and the service health endpoint periodically until
        the job is running and the service is responsive or until timeouts occur.
    """
    start_time = time.time()

    # Check if the Slurm job is running
    while time.time() - start_time < job_timeout:
        # Check if process has exited
        if process and process.poll() is not None:
            logger.error(
                f"Process terminated with exit code {process.returncode} while waiting for job to start"
            )
            return None

        node_address = get_node_address(job_name)
        if node_address:
            logger.info(f"Job {job_name} is running.")
            break
        else:
            logger.info(
                f"Job {job_name} is not running yet. Checking again in {job_interval} seconds..."
            )
        time.sleep(job_interval)
    else:
        logger.error("Job did not start within the timeout period.")
        return None

    # Construct the API URL using the node address
    api_url = api_url_template.format(node_address=node_address)

    # Check if the service is running
    start_time = time.time()
    while time.time() - start_time < service_timeout:
        # Check if process has exited
        if process and process.poll() is not None:
            logger.error(
                f"Process terminated with exit code {process.returncode} while waiting for service to start"
            )
            return None

        try:
            is_service_up = check_service_status(
                api_url, endpoint, api_key, expected_status
            )
        except requests.exceptions.ConnectionError as e:
            logger.info(
                f"Connection error while checking service status: {e}. Service might still be starting up."
            )
            is_service_up = False
        except Exception as e:
            logger.error(f"Unexpected error while checking service status: {e}")
            is_service_up = False

        if is_service_up:
            logger.info("Service is up and running.")
            return api_url
        else:
            logger.info(
                f"Service is not up yet. Checking again in {service_interval} seconds..."
            )
        time.sleep(service_interval)
    else:
        logger.error("Service did not start within the timeout period.")
        return None
