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

from .deploy import deploy_llm, get_node_address, monitor_job_and_service
from .gradio import display_iframe
from .util import enable_logging

__all__ = [
    "monitor_job_and_service",
    "deploy_llm",
    "enable_logging",
    "get_node_address",
    "display_iframe",
]
