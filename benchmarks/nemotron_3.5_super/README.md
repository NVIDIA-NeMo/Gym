# Nemotron 3.5 Super Evaluation setup
- [Nemotron 3.5 Super Evaluation setup](#nemotron-35-super-evaluation-setup)
  - [Run production evals](#run-production-evals)
  - [Development commands](#development-commands)
    - [Build eval container](#build-eval-container)
    - [Launch vLLM](#launch-vllm)
    - [Interactive development on GPUs with Ray cluster](#interactive-development-on-gpus-with-ray-cluster)
    - [Run eval against external vLLM endpoint](#run-eval-against-external-vllm-endpoint)


## Run production evals
TODO @bxyu-nvidia: Will publish these by Thu Jul 30

Results will appear in that checkpoint folder.


## Development commands
### Build eval container
Example run:
```bash
SBATCH_ACCOUNT=my-slurm-account \
SBATCH_PARTITION=batch \
INPUT_CONTAINER=/path/to/vllm/container \
OUTPUT_CONTAINER=/path/to/vllm/container___with_gym.sqsh \
MOUNTS=/path/to/env.yaml:/opt/Gym/env.yaml:x-create=file,/path/to/config.yaml:/opt/Gym/config.yaml:x-create=file \
GYM_CONFIG=benchmarks/nemotron_3.5_super/eval_container_config.yaml \
sbatch --gres=gpu:4 \
  benchmarks/nemotron_3.5_super/build_eval_container.sh
```


Production command (run from root of Gym repo):
```bash
CONTAINER_IMAGE_PATH=vllm/vllm-openai:v0.25.1
mkdir -p "$(dirname "$CONTAINER_IMAGE_PATH")"
enroot import -o "results/$CONTAINER_IMAGE_PATH" "docker://${CONTAINER_IMAGE_PATH}"

SBATCH_ACCOUNT=nemotron_n4_post \
SBATCH_PARTITION=batch \
INPUT_CONTAINER=$(pwd)/results/vllm/vllm-openai:v0.25.1 \
OUTPUT_CONTAINER=$(pwd)/results/vllm/vllm-openai:v0.25.1___with_gym.sqsh \
MOUNTS=$(pwd)/env.yaml:/opt/Gym/env.yaml:x-create=file,$(pwd)/benchmarks/nemotron_3.5_super/eval_container_config.yaml:/opt/Gym/benchmarks/nemotron_3.5_super/eval_container_config.yaml:x-create=file \
GYM_CONFIG=benchmarks/nemotron_3.5_super/eval_container_config.yaml \
sbatch --gres=gpu:4 --qos=interactive benchmarks/nemotron_3.5_super/build_eval_container.sh
```


### Launch vLLM
This script assumes:
- GB200s which are 4 GPUs per node. If you want to use 8 GPUs per node, update the --tensor-parallel-size and --gres=gpu arguments to 8.
- Nemotron 3 Ultra configs e.g. with the parser configs.

Example run:
```bash
MODEL=/path/to/model \
NUM_NODES=4 \
SBATCH_ACCOUNT=my-slurm-account \
SBATCH_PARTITION=batch \
CONTAINER=/path/to/vllm/container \
MOUNTS=/shared/fs:/shared/fs \
bash benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh
```

Production command (run from root of Gym repo):
```bash
MODEL=/lustre/fsw/portfolios/llmservice/users/riship/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
NUM_NODES=4 \
SBATCH_ACCOUNT=nemotron_n4_post \
SBATCH_PARTITION=batch \
SBATCH_QOS=interactive \
CONTAINER=$(pwd)/results/vllm/vllm-openai:v0.25.1___with_gym.sqsh \
MOUNTS=/lustre:/lustre \
bash benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh
```


### Interactive development on GPUs with Ray cluster
Example run:
```bash
NUM_NODES=4 \
SBATCH_ACCOUNT=my-slurm-account \
SBATCH_PARTITION=batch \
SBATCH_GRES=gpu:4 \
CONTAINER=/path/to/vllm/container \
MOUNTS=/shared/fs:/shared/fs \
bash scripts/sbatch_interactive.sh
```

Production command (run from root of Gym repo):
```bash
NUM_NODES=4 \
SBATCH_ACCOUNT=nemotron_n4_post \
SBATCH_PARTITION=batch \
SBATCH_QOS=interactive \
SBATCH_GRES=gpu:4 \
SBATCH_SEGMENT_SIZE=4 \
CONTAINER=$(pwd)/results/vllm/vllm-openai:v0.25.1___with_gym.sqsh \
MOUNTS=/lustre:/lustre \
bash scripts/sbatch_interactive.sh
```


### Run eval against external vLLM endpoint
This script assumes:
- The container is one built via benchmarks/nemotron_3.5_super/build_eval_container.sh
- GB200s which are 4 GPUs per node. If you want to use 8 GPUs per node, update the --tensor-parallel-size and --gres=gpu arguments to 8.
- Nemotron 3 Ultra configs e.g. with the parser configs.

If you want to use your own custom local Gym, please mount:
```bash
MOUNTS=/shared/fs:/shared/fs,/path/to/custom/local/Gym:/opt/Gym
```
The existing Gym venv and individual server venvs will still use the ones baked into the container.

Example run:
```bash
MODEL=/path/to/model \
EXPERIMENT_NAME=my-experiment-name \
NUM_NODES=4 \
SBATCH_ACCOUNT=my-slurm-account \
SBATCH_PARTITION=batch \
CONTAINER=/path/to/vllm/container \
MOUNTS=/shared/fs:/shared/fs \
bash benchmarks/nemotron_3.5_super/sbatch_eval_with_external_vllm.sh \
--config benchmarks/my-benchmark/config.yaml
```

Production run:
```bash
MODEL=/lustre/fsw/portfolios/llmservice/users/riship/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
EXPERIMENT_NAME=gpqa-dev \
NUM_NODES=4 \
SBATCH_ACCOUNT=nemotron_n4_post \
SBATCH_PARTITION=batch \
SBATCH_QOS=interactive \
CONTAINER=$(pwd)/results/vllm/vllm-openai:v0.25.1___with_gym.sqsh \
MOUNTS="/lustre:/lustre,$(pwd)/env.yaml:/opt/Gym/env.yaml" \
bash benchmarks/nemotron_3.5_super/sbatch_eval_with_external_vllm.sh \
--config responses_api_models/vllm_model/configs/vllm_model.yaml \
--config benchmarks/gpqa/config.yaml
```
