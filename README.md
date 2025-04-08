# NERSC Chatbot Deployment

Deploy Hugging Face large language models (LLMs) on NERSC supercomputers using Slurm and the vLLM serving framework. This package supports both command-line interface (CLI) and Python library usage, with utilities for seamless Gradio integration on NERSC JupyterHub.

The deployed models expose an OpenAI-compatible API endpoint powered by vLLM, enabling easy integration with existing OpenAI clients and tools while requiring a secure API key to control and restrict access.

## Quick Start

Install the package:

```bash
module load python
python -m pip install git+https://github.com/asnaylor/nersc_chatbot_deploy
```

Deploy a model using the CLI:

```bash
nersc-chat deploy -A your_account -m meta-llama/Llama-3.1-8B-Instruct
# Use `nersc-chat deploy --help` for more options
```

Or deploy using the Python library:

```python
from nersc_chatbot_deploy import deploy_llm

proc, api_key = deploy_llm(
    account='your_account',
    num_gpus=1,
    queue='shared_interactive',
    time='01:00:00',
    job_name='vLLM_test',
    model='meta-llama/Llama-3.1-8B-Instruct'
)
```

## Features

- Deploy LLMs on NERSC Slurm clusters with GPU allocation
- Monitor Slurm jobs and deployed services
- Embed Gradio UIs inline within Jupyter notebooks on NERSC JupyterHub

## Installation & Prerequisites

- Active NERSC account with permissions to run jobs on Perlmutter
- Optional: Hugging Face access token set as `HF_TOKEN` environment variable for certain models

## Usage

For detailed usage instructions, including CLI options and Python library examples, please refer to the [Docs](docs/README.md) page.

## Security Best Practices

- **Protect API Keys and Tokens:** Always store API keys and tokens securely, preferably in environment variables. Avoid hard-coding them in code or sharing publicly.
- **Use Trusted Models:** Only download and deploy models from reputable and verified sources, such as official Hugging Face repositories or other trusted providers. Verify the integrity and authenticity of the models to avoid potential security risks, such as malicious code or data leaks.
- **Respect Data Privacy:** Avoid using sensitive or personal data unless absolutely necessary, and ensure compliance with data privacy regulations.
- **Follow Licensing and IP Rights:** Comply with all model licenses and institutional policies when deploying and using models.
- **Restrict Access:** Limit access to deployed services by enforcing API key authentication and network-level restrictions where possible.

## Troubleshooting

- Ensure your NERSC account has proper permissions.
- Verify Slurm queue and constraints match your allocation.
- Check logs for errors; adjust `--log-level` for more verbosity.
- Confirm network access for Gradio proxy URLs on JupyterHub.
