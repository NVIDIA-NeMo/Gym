# Nemotron 3.5 Super Evaluation setup
- [Nemotron 3.5 Super Evaluation setup](#nemotron-35-super-evaluation-setup)
  - [Build eval container](#build-eval-container)



## Build eval container
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
