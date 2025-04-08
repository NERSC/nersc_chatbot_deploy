"""
nersc_chatbot_deploy package initialization.

This module initializes the nersc_chatbot_deploy package by importing key functions
for deploying and monitoring LLM models on NERSC, as well as utilities for displaying
Gradio interfaces. It also configures the logging settings based on environment variables.

Exports:
    - deploy_llm: Function to deploy large language models on Slurm clusters.
    - get_node_address: Function to retrieve the node address of a running Slurm job.
    - monitor_job_and_service: Function to monitor Slurm jobs and deployed services.
    - display_iframe: Utility to display an iframe for embedding Gradio UIs.
"""

import logging
import os

from .deploy import deploy_llm, get_node_address, monitor_job_and_service
from .gradio import display_iframe

__all__ = [
    "monitor_job_and_service",
    "deploy_llm",
    "get_node_address",
    "display_iframe",
]

# Configure logging based on LOG_LEVEL environment variable (default: WARNING)
log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
