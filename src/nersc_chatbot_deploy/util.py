"""
Utility functions and enumerations for nersc_chatbot_deploy.

This module provides enums for supported backends and log levels, as well as helper
functions for parsing command-line style strings, generating unique job names, and
dumping job metadata to JSON files.
"""

import json
import logging
import random
import uuid
from enum import Enum
from typing import Dict, Union


class SupportedBackends(str, Enum):
    """Enumeration of supported backend serving frameworks."""

    vllm = "vllm"


class LogLevel(str, Enum):
    """Enumeration of supported logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def parse_string_to_dict(input_string: str) -> Dict[str, str]:
    """
    Converts a specially formatted string into a dictionary.

    The input string is expected to contain key-value pairs separated by spaces,
    where keys are prefixed with '--'. Keys without associated values will have an empty string as the value.

    Args:
        input_string (str): The input string containing key-value pairs.

    Returns:
        Dict[str, str]: A dictionary representing the key-value pairs from the input string.
    """
    # Split the input string into components
    components = input_string.split()

    # Initialize an empty dictionary
    result_dict = {}

    # Iterate through the components and build the dictionary
    i = 0
    while i < len(components):
        if components[i].startswith("--"):
            key = components[i].lstrip("--")  # Remove the leading '--'
            # Check if the next component is also a key or if it exists
            if i + 1 < len(components) and not components[i + 1].startswith("--"):
                value = components[i + 1]  # The next component is the value
                i += 2  # Move to the next key-value pair
            else:
                value = ""  # No value for this key
                i += 1  # Move to the next key
            result_dict[key] = value
        else:
            i += 1  # Move to the next component

    return result_dict


def generate_unique_name(backend: str) -> str:
    """
    Generate a unique name for a job by combining a random adjective, surname, and a shortened UUID.

    Args:
        backend (str): A string to be included in the generated name.

    Returns:
        str: A unique, memorable name for the job.
    """
    adjectives = ["quick", "lazy", "sleepy", "noisy", "hungry"]
    surnames = ["Smith", "Johnson", "Williams", "Jones", "Brown"]
    adjective: str = random.choice(adjectives)
    surname: str = random.choice(surnames)
    unique_id: str = str(uuid.uuid4())[:4]  # Shorten UUID to 8 characters
    return f"{backend}_{adjective}_{surname}_{unique_id}"


def dump_job_data_to_json(
    file_path: str, job_name: str, llm_address: str, model: str, api_key: str
) -> None:
    """
    Dumps job data to a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        job_name (str): The name of the job.
        llm_address (str): The address of the LLM.
        model (str): The LLM model name.
        api_key (str): The API key.
    """
    data = {
        "job_name": job_name,
        "llm_address": llm_address,
        "model": model,
        "api_key": api_key,
    }

    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def enable_logging(level: Union[int, str] = logging.DEBUG) -> None:
    """
    Enable console logging for interactive use in Jupyter notebooks and CLI tools.

    Args:
        level: The logging level to set (default: logging.DEBUG).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Prevents duplicate handlers and overwrites existing config
    )
