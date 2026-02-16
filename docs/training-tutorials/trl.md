(training-trl)=

# RL Training with TRL

[TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) is Hugging Face's library for post-training foundation models. This integration enables training models in NeMo Gym environments using TRL's GRPOTrainer with vLLM server mode.

### Install TRL and NeMo Gym

1. **Install TRL venv with vLLM and some extras**

   ```bash
   cd trl/
   uv venv
   source .venv/bin/activate
   uv sync --extra vllm
   uv pip install fastapi uvicorn accelerate deepspeed wandb omegaconf
   ```

1. **Install NeMo Gym in a separate venv**

   ```bash
   git clone https://github.com/NVIDIA-NeMo/Gym.git
   cd Gym
   uv venv --python 3.12
   source .venv/bin/activate
   uv sync
   ```

### Prepare a Dataset

In this example we use the reasoning gym resources server in NeMo Gym to train a model in sudoku:

```bash
cd Gym
source .venv/bin/activate
uv pip install reasoning-gym
cd resources_servers/reasoning_gym
python scripts/create_dataset.py \
    --task mini_sudoku \
    --size 2000 \
    --seed 42 \
    --output data/reasoning_gym/train_mini_sudoku.jsonl

python scripts/create_dataset.py \
    --task mini_sudoku \
    --size 50 \
    --seed 24 \
    --output data/reasoning_gym/val_mini_sudoku.jsonl
```

## Interactive Training

Training requires 2+ GPUs, one for the vLLM server, and one for training. The NeMo Gym TRL integration currently depends on vLLM server mode.

To run training on a single node, launch the NeMo Gym servers, vLLM server, then run training:

### Setup

1. **Update Environment Config**

   Update `env.yaml` in `Gym/` to include model information:

   ```yaml
   policy_base_url: http://127.0.0.1:8000/v1
   policy_api_key: EMPTY
   policy_model_name: Qwen/Qwen2.5-1.5B-Instruct
   ```

2. **Update Training Config**

   Update `examples/scripts/nemo_gym/config.yaml` to point to the mini sudoku dataset:

   ```yaml
   model_name: "Qwen/Qwen2.5-1.5B-Instruct"

   dataset_path: "/path/to/Gym/resources_servers/reasoning_gym/data/reasoning_gym/train_mini_sudoku.jsonl"
   eval_dataset_path: "/path/to/Gym/resources_servers/reasoning_gym/data/reasoning_gym/val_mini_sudoku.jsonl"

   task: "mini-sudoku"
   output_dir: "outputs/nemo_gym_sudoku"

   learning_rate: 1.0e-5
   num_generations: 16
   per_device_train_batch_size: 8
   gradient_accumulation_steps: 1
   max_completion_length: 10000
   vllm_importance_sampling_correction: true

   temperature: 1.0
   top_p: 0.999
   ```

### Run Training


1. **Start NeMo Gym Servers**

   ```bash
   cd Gym/
   source .venv/bin/activate

   config_paths="resources_servers/reasoning_gym/configs/reasoning_gym.yaml,\
   responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"

   ng_run "+config_paths=[${config_paths}]"
   ```

1. **Start TRL vLLM Server on GPU 0**

   ```bash
   cd trl/
   source .venv/bin/activate
   CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
     --model Qwen/Qwen2.5-1.5B-Instruct \
     --max-model-len 16384 \
     --host 0.0.0.0 \
     --port 8000
   ```

1. **Run Training on GPU 1**

   ```bash
   cd trl/
   source .venv/bin/activate
   cd examples/scripts/nemo_gym

   CUDA_VISIBLE_DEVICES=1 python train_multi_environment.py --config config.yaml
   ```

## Multi-Node Training with Slurm

An example five-node training script is provided in `submit.sh`. Nodes one through four run the training backend, while node five runs vLLM inference for NeMo Gym agent rollouts.

Before running the Slurm script, ensure you have completed the TRL and NeMo Gym installation steps above. The script assumes `.venv` directories exist for both TRL and Gym. If you use a container in the Slurm script, you should also create the virtual environments from the container in an interactive session or with a separate sbatch script.

1. **Configure the Script**

   Update `submit.sh` with your Slurm account, partition, paths to your project directory, and updated training configs.

1. **Submit the Job**

   ```bash
   sbatch submit.sh
   ```

1. **Monitor Training**

   ```bash
   tail -f logs/<job_id>/*
   ```

## Multi-Environment Training

NeMo Gym is designed to enable training on many environments simultaneously and at scale. This allows learning diverse capabilities, such as tool calling and reasoning, in a single training run. In this example, we add the workplace assistant environment to the mini sudoku setup above, which is a multi-step tool use environment for office tasks.

1. **Prepare Workplace Assistant Dataset**

   Many NeMo Gym datasets used to train Nemotron models are available on Hugging Face. Use `ng_prepare_data` to download and prepare datasets. This command:

   - Downloads the dataset from Hugging Face
   - Validates the format and computes metrics
   - Adds an `agent_ref` field to each example that tells NeMo Gym which agent server should handle that example

   First, create `env.yaml` in `Gym/` with your HF token:

   ```yaml
   hf_token: <your_hf_token>
   ```

   Then prepare the dataset:

   ```bash
   cd Gym
   source .venv/bin/activate

   config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
   resources_servers/workplace_assistant/configs/workplace_assistant.yaml"

   ng_prepare_data "+config_paths=[${config_paths}]" \
       +output_dirpath=data/workplace_assistant \
       +mode=train_preparation \
       +should_download=true \
       +data_source=huggingface
   ```

   This creates `train.jsonl` and `validation.jsonl` files in `data/workplace_assistant/`.

1. **Create Combined Dataset**

   Combine datasets into a single file with tasks from both environments:

   ```bash
   cat data/workplace_assistant/train_workplace.jsonl data/reasoning_gym/train_mini_sudoku.jsonl | shuf > train_multi_env.jsonl
   ```

   > **Tip**: Ensure datasets are the same size before shuffling for an even blend of tasks. Repeat for the validation dataset.

1. **Update Training Config**

   Update the config to point to the combined dataset:

   ```yaml
   model_name: "Qwen/Qwen3-4B-Instruct-2507"

   dataset_path: "/path/to/data/train_multi_env.jsonl"
   eval_dataset_path: "/path/to/data/val_multi_env.jsonl"

   task: "workplace-sudoku"                    # used in wandb run name
   output_dir: "outputs/nemo_gym_multi_env"

   # ... rest of config same
   ```

1. **Update ng_run**

   Whether training interactively or via Slurm, update the `ng_run` command to include config files from each resources server:

   ```bash
   cd Gym
   source .venv/bin/activate

   config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
   resources_servers/workplace_assistant/configs/workplace_assistant.yaml,\
   resources_servers/reasoning_gym/configs/reasoning_gym.yaml"

   ng_run "+config_paths=[${config_paths}]"
   ```

   This starts servers for both environments. The training script automatically routes each example to the correct agent server based on its `agent_ref` field.

1. **Run Training**

   Update the Slurm submission script to use the new training config and both `ng_run` resources server configs, then submit the job as before.

   The training script reads `agent_ref` from each example's metadata, routes requests to the correct NeMo Gym agent server, and handles different agents and environments in the same batch.

## Resources

- [TRL GitHub](https://github.com/huggingface/trl)
- [TRL Documentation](https://huggingface.co/docs/trl/en/index)
