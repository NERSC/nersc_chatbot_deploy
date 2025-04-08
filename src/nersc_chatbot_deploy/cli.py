"""
Command-line interface for nersc_chatbot_deploy.

This module provides a Typer-based CLI to deploy large language models (LLMs) on NERSC
using Slurm and the specified backend (default: vLLM). It supports configuring job parameters,
monitoring deployment status, and optionally dumping deployment credentials to a JSON file.

Commands:
    deploy: Deploy an LLM model with specified resources and options.
"""

import logging
from typing import Dict

import typer
from typing_extensions import Annotated

from nersc_chatbot_deploy.deploy import deploy_llm, monitor_job_and_service
from nersc_chatbot_deploy.util import (
    LogLevel,
    SupportedBackends,
    dump_job_data_to_json,
    generate_unique_name,
    parse_string_to_dict,
)

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def deploy(
    account: Annotated[
        str,
        typer.Option(
            "--account", "-A", help="The account name to be used with the `-A` option."
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="The model to be served by the backend."),
    ],
    num_gpus: Annotated[
        int,
        typer.Option(
            "--num-gpus",
            "-G",
            help="The number of GPUs to request with the `-G` option.",
        ),
    ] = 1,
    queue: Annotated[
        str,
        typer.Option("--queue", "-q", help="The queue to use with the `-q` option."),
    ] = "shared_interactive",
    time: Annotated[
        str,
        typer.Option("--time", "-t", help="The time to allocate with the `-t` option."),
    ] = "01:00:00",
    job_name: Annotated[
        str,
        typer.Option(
            "--job-name", "-j", help="The job name to be used with the `-J` option."
        ),
    ] = "",
    backend: Annotated[
        SupportedBackends, typer.Option("--backend", "-b", help="The backend to use.")
    ] = SupportedBackends.vllm,
    backend_args: Annotated[
        str,
        typer.Option(
            "--backend-args",
            help='Additional arguments to pass to the backend. (Example "--tensor-parallel-size 4").',
        ),
    ] = "",
    constraint: Annotated[
        str,
        typer.Option(
            "--constraint", "-C", help="The constraint to use with the `-C` option."
        ),
    ] = "gpu",
    dump_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Dump Server Address and api-key in a JSON file.",
            is_flag=True,
        ),
    ] = False,
    log_level: Annotated[
        LogLevel, typer.Option("--log-level", "-l", help="Set the logging level.")
    ] = LogLevel.WARNING,
) -> None:
    """
    Deploys the LLM using the specified backend with the provided parameters.

    Args:
        account (str): NERSC account name for Slurm job submission.
        model (str): Hugging Face model identifier to serve.
        num_gpus (int): Number of GPUs to allocate for the job.
        queue (str): Slurm queue/partition to submit the job to.
        time (str): Walltime allocation for the job.
        job_name (str): Optional Slurm job name; auto-generated if empty.
        backend (SupportedBackends): Backend to use for serving the model.
        backend_args (str): Additional backend-specific arguments as a string.
        constraint (str): Slurm constraint for node selection (default "gpu").
        dump_json (bool): Whether to dump deployment info to a JSON file.
        log_level (LogLevel): Logging verbosity level.

    Raises:
        typer.Exit: Exits with code 1 if deployment fails or times out.
    """
    # Set the logging level based on the provided log_level option
    logger.setLevel(getattr(logging, log_level.value))
    logger.info(
        f"Starting deployment with model={model}, account={account}, num_gpus={num_gpus}, job_name={job_name or 'auto-generated'}"
    )

    # Convert string to dictionary
    backend_args_dict: Dict[str, str] = parse_string_to_dict(backend_args)
    logger.debug(f"Parsed backend_args: {backend_args_dict}")

    # Generate a job name if required
    if not job_name:
        job_name = generate_unique_name(backend.value)
    typer.echo(
        f"🚀 Starting deployment of model '{model}' with job name '{job_name}'..."
    )

    # Execute Slurm job and deploy LLM
    try:
        process, llm_api_key = deploy_llm(
            account,
            num_gpus,
            queue,
            time,
            job_name,
            model,
            backend.value,
            backend_args_dict,
            constraint,
        )
        logger.info(
            f"Deployment process started with PID: {process.pid if process else 'None'}"
        )

        # Inform the user that the service is starting up
        typer.echo("⏳ Please wait a few minutes while the service is starting up...")

        # Use monitor_job_and_service to wait for job and service readiness
        LLM_address = monitor_job_and_service(
            job_name=job_name,
            api_url_template="http://{node_address}:8000/v1",
            endpoint="/models",
            api_key=llm_api_key,
            expected_status=200,
            job_timeout=600,  # optional: adjust timeouts as needed
            service_timeout=600,
            job_interval=30,
            service_interval=30,
        )

        if LLM_address is None:
            logger.error("Failed to detect running job or service. Exiting.")
            typer.echo("❌ Error: Deployment failed or timed out.")
            if process:
                process.terminate()
            raise typer.Exit(code=1)

        # Dump address and key to file if requested
        if dump_json:
            dump_job_data_to_json(
                f"{job_name}_creds.json", job_name, LLM_address, model, llm_api_key
            )
            logger.info(f"Credentials dumped to {job_name}_creds.json")
            typer.echo(f"📄 Credentials dumped to {job_name}_creds.json")
        else:
            typer.echo(f"🌐 Access {model} at {LLM_address}")
            typer.echo(f"🔑 Your API key is: {llm_api_key}")

        # Wait for the process to complete, allowing for manual termination
        if process is not None:
            typer.echo("Press Ctrl+C to exit and shut down the LLM.")
            process.wait()
        logger.info("Deployment process ended.")
        typer.echo("✅ Deployment process completed successfully.")

    except KeyboardInterrupt:
        # Handle manual interruption and clean up
        logger.warning("KeyboardInterrupt received. Shutting down the LLM.")
        typer.echo("\n⚠️  Shutting down the LLM.")
        if process:
            process.terminate()


if __name__ == "__main__":
    app()
