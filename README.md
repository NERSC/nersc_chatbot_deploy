# NERSC Chatbot Deployment

This repository provides instructions for deploying a Hugging Face LLM model using vLLM the Slurm-based cluster system at NERSC.


## Prerequisites
1. Hugging Face Access Token: For models that require special access, create a user access token. Follow the instructions [here](https://huggingface.co/docs/hub/en/security-tokens) to create a Hugging Face access token. Update the [HF_TOKEN](deploy.sh#L11)  in deploy.sh with your token.

2. NERSC Account: Ensure you have an active NERSC account and the necessary permissions to run jobs on the Perlmutter system.

## Usage

### Submitting a Job via `sbatch`
To submit a job using `sbatch`, run the following command:
```bash
sbatch -A <account_name> deploy.sh meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 4
```
> [!NOTE]  
> Replace <account_name> with your NERSC account name and add any other relevant slurm or vllm arguments.


### Running Interactively with `salloc`
To run the job interactively, use:
```bash
salloc -N 1 -C gpu -G 4 -t 01:00:00 -q interactive -A <account_name>
./deploy.sh meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 4
```

### Retrieving the Address
To get the address of the running job, execute:
```bash
JOBID=$(squeue --me -h -o "%.18i" | tail -n1)
export ADDRESS=$(scontrol show job $JOBID --json | jq -r '.jobs[0].nodes').chn.perlmutter.nersc.gov
```

### Testing the Server
To test the server, use the following `curl` command:
```bash
curl -X POST "${ADDRESS}:8000/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "meta-llama/Llama-3.3-70B-Instruct",
		"messages": [
			{
				"role": "user",
				"content": "What is the capital of France?"
			}
		]
	}'
```

### Running the Client
To interact with the deployed model programmatically, you can use the following Python script:
```python
import os
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = f"http://{os.getenv('ADDRESS')}:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me how to run a slurm job."},
    ]
)
print("Chat response:", chat_response)
```