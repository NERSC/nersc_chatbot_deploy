# Documentation

This documentation provides detailed instructions for using the `nersc_chatbot_deploy` package, including the command-line interface (CLI), Python library functions, and embedding Gradio UIs on NERSC JupyterHub.

For a high-level overview and quick start, see the [main README](../README.md).

## How It Works

The `nersc_chatbot_deploy` package deploys vLLM within a **Shifter container** on the NERSC Slurm cluster:

1. **Slurm Allocation**: Requests GPU resources using `salloc`
2. **Shifter Container**: Runs vLLM inside a containerized environment
3. **vLLM Service**: Serves the model with an OpenAI-compatible API endpoint
4. **Monitoring**: Tracks job status and service health

**Default Shifter image**: `vllm/vllm-openai:v0.7.3`

## NERSC Environment Setup

**Important:** Configure your environment before deploying:

```bash
# Required for gated models (Llama, Mistral, etc.)
export HF_TOKEN=<your-hf-token>

# Use SCRATCH for better performance and space  
export HF_HOME=$SCRATCH/huggingface

# Optional: Use custom vLLM Shifter image
export vLLM_IMAGE=vllm/vllm-openai:v0.10.0
shifterimg pull $vLLM_IMAGE  # Ensure image is available
```

## Command-Line Interface (CLI)

Use the `nersc-chat` CLI command to deploy models easily. Example:

```bash
export HF_TOKEN=<my-token>
export HF_HOME=$SCRATCH/huggingface
nersc-chat -A your_account -m meta-llama/Llama-3.1-8B-Instruct
```

When the service is up, the CLI will output the service address and API key to stdout. Optionally, you can use the `--json` flag to dump this information to a JSON file for easier programmatic access.

### CLI Options

- `--account`, `-A` (required): NERSC account name for Slurm job submission
- `--model`, `-m` (required): Hugging Face model identifier
- `--num-gpus`, `-G` (default: 1): Number of GPUs to allocate
- `--queue`, `-q` (default: `shared_interactive`): Slurm queue
- `--time`, `-t` (default: `01:00:00`): Walltime allocation
- `--job-name`, `-j`: Optional Slurm job name (auto-generated if omitted)
- `--backend`, `-b` (default: `vllm`): Backend serving framework
- `--backend-args`: Additional backend-specific arguments as a string (Example "--tensor-parallel-size 4")
- `--constraint`, `-C` (default: `gpu`): Slurm node constraint
- `--json`: Dump deployment info to a JSON file
- `--log-level`, `-l` (default: `WARNING`): Logging verbosity level
- `--help`: Show help message

## Python Library

Use the following key functions from the `nersc_chatbot_deploy` package:

- [`deploy_llm`](../src/nersc_chatbot_deploy/deploy.py#L108): Deploy a model programmatically
- [`get_node_address`](../src/nersc_chatbot_deploy/deploy.py#L215): Retrieve the node address of a running job
- [`monitor_job_and_service`](../src/nersc_chatbot_deploy/deploy.py#L294): Monitor job and service status
- [`display_iframe`](../src/nersc_chatbot_deploy/gradio.py#L10): Embed Gradio UI inline in Jupyter notebooks

### Example: Deploying a Model

```python
import os
from nersc_chatbot_deploy import deploy_llm

os.environ['HF_TOKEN'] = "my_token"
os.environ['HF_HOME'] = os.path.join(os.environ.get('SCRATCH'), 'huggingface')

proc, llm_api_key = deploy_llm(
    account='your_account',
    num_gpus=1,
    queue='shared_interactive',
    time='01:00:00',
    job_name='vLLM_test',
    model='meta-llama/Llama-3.1-8B-Instruct'
)
print(f"API key: {llm_api_key}")
```

### Example: Retrieving Service Address

```python
from nersc_chatbot_deploy import get_node_address

node_address = get_node_address('vLLM_test')
if node_address:
    print(f"Service address: http://{node_address}:8000/v1")
else:
    print("Failed to retrieve the service address.")
```

### Example: Monitoring Job and Service Readiness

Use `monitor_job_and_service` to wait for the Slurm job to start and the service to be up:

```python
from nersc_chatbot_deploy import monitor_job_and_service

LLM_address = monitor_job_and_service(
    job_name='vLLM_test',
    api_url_template="http://{node_address}:8000/v1",
    endpoint="/models",
    api_key=llm_api_key,
    expected_status=200,
    job_timeout=600, 
    service_timeout=600,
    job_interval=30,
    service_interval=30,
)

if LLM_address:
    print(f"LLM service is up at: {LLM_address}")
else:
    print("Failed to start the LLM service.")
```

## Accessing the LLM via Gradio on NERSC JupyterHub

Launch and embed the Gradio chat interface inline within a Jupyter notebook:

```python
import gradio as gr
import os
from nersc_chatbot_deploy import display_iframe

port = 8989  # Or any available port
root_path = f"{os.environ['JUPYTERHUB_SERVICE_PREFIX']}proxy/{port}/"
proxy_url = f"https://jupyter.nersc.gov{root_path}"

# Replace `node_address` and `llm_api_key` with your actual values
gr.load_chat(f"http://{node_address}:8000/v1", model="meta-llama/Llama-3.1-8B-Instruct", token=llm_api_key).launch(
    server_name="0.0.0.0", server_port=port, share=False, root_path=root_path, inline=False
)

display_iframe(proxy_url)
```

## Multi-node Deployments

Currently, the `nersc_chatbot_deploy` package does not support multi-node deployments through either the command-line interface (CLI) or the Python library. All deployments are limited to single-node configurations, utilizing 4 Nvidia A100 GPUs per node. However, the underlying vLLM serving framework does support multi-node deployments using Ray.

For users interested in setting up multi-node deployments manually, you can refer to the [example Slurm script for multi-node deployment](multinode_vllm.sh) that demonstrates how to configure a multi-node vLLM deployment using Ray.

This script provides a basic framework for deploying vLLM across multiple nodes using Ray, leveraging the capabilities of NERSC's compute resources.

**Note:** Keep your API key (`llm_api_key`) secure and do not expose it publicly.

## Troubleshooting

- Ensure your NERSC account has proper permissions.
- Verify Slurm queue and constraints match your allocation.
- Check logs for errors; adjust `--log-level` for more verbosity.
- Confirm network access for Gradio proxy URLs on JupyterHub.
