#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=4
#SBATCH --qos=debug

#Environment Variables
export HF_TOKEN=$(cat $HOME/.hf_token)
export HF_HOME=$SCRATCH/huggingface/
export VLLM_VERSION="v0.7.3"
export VLLM_IMAGE="vllm/vllm-openai:${VLLM_VERSION}"

# Check for required Hugging Face model argument
if [ $# -eq 0 ]; then
  echo "Usage: $0 <Hugging Face model>"
  exit 1
fi

# Print the model being used for clarity
echo "Deploying model: $1"

# Run the vllm serve command within Shifter
srun \
    shifter \
        --image=$VLLM_IMAGE \
        --module=gpu,nccl-plugin \
            vllm serve "$@"