#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --constraint='gpu&hbm40g'
#SBATCH --gpus=4

#SBTACH --time=01:00:00
#SBATCH --qos=regular

#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128

# Variables
export HF_TOKEN=`cat ~/.hf_token`
export HF_HOME=$SCRATCH/huggingface 
VLLM_IMAGE="vllm/vllm-openai:v0.8.3"
SHIFTER="shifter --image=$VLLM_IMAGE --module=gpu,nccl-plugin  --env PYTHONUSERBASE=${SCRATCH}/vllm_v0.8.3 "
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=( $nodes )

# Ray Head Node
echo "<> Starting ray head...."
srun --nodes=1 --ntasks=1 -w ${nodes_array[0]} --unbuffered \
    $SHIFTER ray start --head --block &

sleep $((1 * 60))

# Ray Worker Nodes
worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
echo "<> Starting ${worker_num} ray workers..."

ray_head_node="${nodes_array[0]}"
for ((  i=1; i<=$worker_num; i++ )); do
    node_i=${nodes_array[$i]}
    echo "    - $i at $node_i"
    srun --nodes=1 --ntasks=1 -w $node_i --unbuffered \
        $SHIFTER ray start --address "${ray_head_node}:6379" --block &
done

# Check if all workers are active
ray_init_timeout=300
ray_cluster_size=$SLURM_JOB_NUM_NODES
for (( i=0; i < $ray_init_timeout; i+=5 )); do
    active_nodes=`$SHIFTER python3 -c 'import ray; ray.init(); print(sum(node["Alive"] for node in ray.nodes()))'`
    if [ $active_nodes -eq $ray_cluster_size ]; then
        echo "All ray workers are active and the ray cluster is initialized successfully."
        break
    fi
    echo "Wait for all ray workers to be active. $active_nodes/$ray_cluster_size is active"
    sleep 5s;
done

# Launch vLLM serve
$SHIFTER vllm serve "meta-llama/Llama-3.3-70B-Instruct" --tensor-parallel-size 8