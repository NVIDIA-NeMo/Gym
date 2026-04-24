(training-verl)=

# Training with verl

This tutorial shows how to run RL training on NeMo Gym environments using the `nemo_gym` recipe in [verl](https://github.com/verl-project/verl).

**verl** is a flexible, efficient RL training framework. The NeMo Gym integration is a recipe in the [verl-recipe](https://github.com/verl-project/verl-recipe) submodule under `recipe/nemo_gym/`, and is tested with vLLM 0.17 (`verlai/verl:vllm017.latest`).

## 1. Prepare training data

Using NeMo Gym, prepare the training dataset for your environment. Each row needs an `agent_ref` field so NeMo Gym can route it to the right agent:

```bash
cd $NEMO_GYM_ROOT
source .venv/bin/activate

config_paths="resources_servers/workplace_assistant/configs/workplace_assistant.yaml,\
responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"

ng_prepare_data \
    "+config_paths=[${config_paths}]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
```

This produces `data/workplace_assistant/{train,validation}.jsonl` ready for training.

## 2. Set environment variables

In your verl clone, copy the recipe's `config.env.example` and fill in your paths:

```bash
cd $VERL_ROOT
cp recipe/nemo_gym/config.env.example config.env
```

```bash
# config.env
VERL_ROOT=/path/to/verl
NEMO_GYM_ROOT=/path/to/nemo-gym
HF_HOME=/path/to/hf_home
RESULTS_ROOT=/path/to/results
WANDB_USERNAME=your_username
WANDB_API_KEY=your_key
```

## 3. Point verl at NeMo Gym

Each training run needs a YAML listing the NeMo Gym servers to launch (see `recipe/nemo_gym/configs/` for examples):

```yaml
# recipe/nemo_gym/configs/workplace.yaml
nemo_gym:
  nemo_gym_root: $NEMO_GYM_ROOT
  uses_reasoning_parser: false         # set true for reasoning models
  config_paths:
    - $NEMO_GYM_ROOT/responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
    - $NEMO_GYM_ROOT/resources_servers/workplace_assistant/configs/workplace_assistant.yaml
```

The first config launches the model server. Each additional entry adds an environment. For multi-environment training, list every environment here and make sure your training JSONL has the matching `agent_ref`s.

## 4. Use the recipe when launching verl training

In your verl training script, swap in the NeMo Gym dataset loader and agent-loop manager:

```bash
+data.custom_cls.path=recipe/nemo_gym/dataset.py
+data.custom_cls.name=NeMoGymJSONLDataset
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.nemo_gym.agent_loop.NeMoGymAgentLoopManager
+actor_rollout_ref.rollout.agent.agent_loop_config_path=${VERL_ROOT}/recipe/nemo_gym/configs/workplace.yaml
```

## 5. Launch

The recipe includes example Slurm job submission scripts (`submit_math.sh`, `submit_workplace.sh`, `submit_multienv.sh`). Update these with your slurm specific variables such as account and partition, then submit:

```bash
cd $VERL_ROOT
sbatch recipe/nemo_gym/submit_workplace.sh
```

---

For additional details, see [`recipe/nemo_gym/README.rst`](https://github.com/verl-project/verl-recipe/blob/main/nemo_gym/README.rst).
