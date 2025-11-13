# RL Training with NeMo RL

**Goal**: Train a model with NeMo RL.

Multinode Slurm script and run command are found at the bottom of this document. Please do the single node setup first\!

## Single GPU node setup to ensure correctness

### Request an interactive node

Please use your local Mac or an interactive GPU node on the cluster to try Gym only out. The login and vscode nodes on the cluster are not designed for this kind of workload and you will run into many random issues. CPU nodes are ok, but they are also kind of weird and sometimes surprisingly slow.

```shell
# DFW single node interactive
srun -A llmservice_modelalignment_sft -p interactive \
    --no-container-mount-home \
    --container-mounts=/lustre:/lustre \
--container-image=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/nvcr.io/nvidian/nemo-rl:bd2e645-37644239.squashfs \
    --gres=gpu:8 \
    --nodes=1 --ntasks=1 --time 04:00:00 \
    --pty /bin/bash

# NRT single node interactive
srun -A llmservice_modelalignment_sft -p interactive \
    --no-container-mount-home \
    --container-mounts=/lustre:/lustre \
    --container-image=/lustre/fsw/portfolios/llmservice/users/abhibhag/.cache/containers/nemo-rl:bd2e645-37644239.squashfs \
    --gres=gpu:8 \
    --nodes=1 --ntasks=1 --time 04:00:00 \
    --pty /bin/bash
```

### Setup

```shell
# CD into your preferred workspace
# cd /lustre/fsw/portfolios/llmservice/users/$USER/...

# Clone the private repository
git clone https://gitlab-master.nvidia.com/nexus-team/nemo-rl
cd nemo-rl
git checkout bxyu/nemo-gym-integration-main

# Clone Penguin
git clone https://{your NV Github username}:{your NV Github PAT}@github.com/NVIDIA-NeMo/Gym.git 3rdparty/Penguin-workspace/Penguin

# Pull necessary submodules (e.g. megatron, automodel, etc). Nothing Gym-specific.
git submodule update --init --recursive

# Initial setup
source /opt/nemo_rl_venv/bin/activate
uv sync --group={build,docs,dev,test} --extra penguin

# This will take 10-15 mins
# NOTE: the test `unit/environments/test_penguin.py::test_penguin_sanity` is failing with a Ray error and currently being worked on in this PR. This has no impact on the training!
# We add the HF token here to avoid HF rate limits
HF_HOME=.cache/ \
HF_TOKEN={your HF token} \
    ./examples/penguin/run_penguin_single_node_sanity_tests.sh

# If you used Gym previously, to run these tests properly, you may need to set `NRL_FORCE_REBUILD_VENVS=true` on an initial run or something.
# If you've run these tests before and are getting HF rate limit errors, you can add `HF_HUB_OFFLINE=1`
```

### Prepare NeMo Gym data

You will need to all `ng_prepare_data` for all data you intend to train on, including data that you already have locally. The `ng_prepare_data` command will add an agent\_ref property to each example that tells Gym which agent to route that example to\!

The `ng_prepare_data` command below includes a model config, not because a model is used in the preparation step, but in order to help Gym resolve the agent config properly. Gym will error out if it detects anything wrong about the config. Specifically in this case, an agent config will require a model reference and it will error out if a model ref is not provided.

```shell
# Setup Penguin local venv
cd 3rdparty/Penguin-workspace/Penguin
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
uv sync --active --extra dev

# Add your Gitlab info into the local env.yaml if you have not already
echo "mlflow_tracking_uri: https://gitlab-master.nvidia.com/api/v4/projects/191584/ml/mlflow/
mlflow_tracking_token: {your Gitlab PAT}" > env.yaml

# Prepare data
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml"
ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/bytedtsinghua_dapo17k \
    +mode=train_preparation +should_download=true

# Return to NeMo RL directory and Python env
cd ../../.. && source /opt/nemo_rl_venv/bin/activate
```

### Single node training

```shell
# Run example training config for single node
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"
EXP_NAME="$(date +%Y%m%d)/penguin_grpo/qwen3_4binstruct/dapo17k_bytedtsinghua_test_001"
CONFIG_PATH=examples/penguin/grpo_dapo17k_bytedtsinghua_qwen3_4binstruct_nf.yaml
HF_HOME=.cache/ \
WANDB_API_KEY={your W&B API key} \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="{your NV username}-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    grpo.val_at_start=false \
    ++grpo.num_prompts_per_step=4 \
    ++grpo.max_num_steps=3 \
    ++policy.dtensor_cfg.clear_cache_every_n_steps=1 \
    ++cluster.num_nodes=1 \
    checkpointing.checkpoint_dir=results/$EXP_NAME &

```

## Multi node

### Submit script

i.e. `temp_penguin_submit.sh` or smth in the root NeMo RL dir.

```shell
# ----- PARAMETERS -----
# WANDB_API_KEY, SLURM_ACCOUNT, SLURM_PARTITION, EXP_NAME, NUM_ACTOR_NODES, REPO_LOCATION

# ----- CONSTANTS -----
CONTAINER_IMAGE_PATH=/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/bxyu/cache/containers/nvcr.io/nvidian/nemo-rl:8762f57-37992537

read -r -d '' COMMAND <<EOF
cd ${REPO_LOCATION}

HF_HOME=.cache/ \
HF_HUB_OFFLINE=1 \
WANDB_API_KEY=$WANDB_API_KEY \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    cluster.num_nodes=$NUM_ACTOR_NODES \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    checkpointing.checkpoint_dir=results/$EXP_NAME \
    $@
EOF

echo -e "Running command:\n$COMMAND"

# Not sure why this is necessary, but ray.sub needs to be launched from the NeMo-RL root directory
cd $REPO_LOCATION
COMMAND=$COMMAND \
CONTAINER=$CONTAINER_IMAGE_PATH \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=$NUM_ACTOR_NODES \
    --account=$SLURM_ACCOUNT \
    --partition=$SLURM_PARTITION \
    --time=4:0:0 \
    --job-name=$EXP_NAME \
    --gres=gpu:8 \
    ray.sub
```

### Submit script run

```shell
WANDB_API_KEY={your W&B API key} \
SLURM_ACCOUNT=llmservice_modelalignment_sft \
SLURM_PARTITION=batch \
EXP_NAME=penguin_grpo/qwen3_4binstruct/8nodes/dapo17k_bytedtsinghua_nf_001 \
NUM_ACTOR_NODES=8 \
REPO_LOCATION={your NeMo RL dir}\
    ./temp_penguin_submit.sh \
    --config=examples/penguin/grpo_dapo17k_bytedtsinghua_qwen3_4binstruct_nf.yaml \
    logger.wandb.project="$USER-nemo-gym-rl-integration"
```

## Other verifiers

### Comp coding

Prepare data

```shell
config_paths="resources_servers/comp_coding/configs/comp_coding.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/comp_coding \
    +mode=train_preparation +should_download=true
```

Single node run

```shell
pkill -f VllmAsyncGenerationWorker
uv run ray stop --force
uv run python -c "import ray; ray.shutdown()"
EXP_NAME="$(date +%Y%m%d)/penguin_grpo/qwen3_4binstruct/comp_coding_test_001"
CONFIG_PATH=examples/penguin/grpo_comp_coding_qwen3_4binstruct.yaml
HF_HOME=.cache/ \
WANDB_API_KEY={your W&B key} \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="{your NV username}-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    ++grpo.val_at_start=false \
    ++grpo.num_prompts_per_step=4 \
    ++grpo.max_num_steps=3 \
    ++policy.dtensor_cfg.clear_cache_every_n_steps=1 \
    ++cluster.num_nodes=1 \
    checkpointing.checkpoint_dir=results/$EXP_NAME &
```

Multi node run

```shell
WANDB_API_KEY={your W&B API key} \
SLURM_ACCOUNT=llmservice_modelalignment_sft \
SLURM_PARTITION=batch \
EXP_NAME=20250926/penguin_grpo/qwen3_4binstruct/8nodes/comp_coding_001 \
NUM_ACTOR_NODES=8 \
REPO_LOCATION={your NeMo RL dir} \
    ./temp_penguin_submit.sh \
    --config=examples/penguin/grpo_comp_coding_qwen3_4binstruct.yaml \
    logger.wandb.project="$USER-nemo-gym-rl-integration"
```

### Ext Taubench Env

Provide user model credentials

```shell
echo "ext-taubench-env_user_model:
  responses_api_models:
    vllm_model:
      base_url: https://integrate.api.nvidia.com/v1
      api_key: <Reach out to @bxyu or @banghuaz for this API key>
      model: private/nemotron/qwen/qwen3-235b-a22b-instruct-2507" > penguin_env.yaml
```

Prepare data in the Penguin directory and venv

```shell
config_paths="responses_api_agents/ext-taubench-env-agent/configs/ext-taubench-env.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/ext-taubench-env \
    +mode=train_preparation \
    +should_download=true \
    '+ext-taubench-env_user_model.responses_api_models.vllm_model.base_url=dummy' \
    '+ext-taubench-env_user_model.responses_api_models.vllm_model.api_key=dummy' \
    '+ext-taubench-env_user_model.responses_api_models.vllm_model.model=dummy'
```

Single node run

```shell
pkill -f VllmAsyncGenerationWorker
uv run ray stop --force
ffuv run python -c "import ray; ray.shutdown()"
EXP_NAME="$(date +%Y%m%d)/penguin_grpo/qwen3_4binstruct/ext_taubench_env_test_001"
CONFIG_PATH=examples/penguin/grpo_ext_taubench_env_qwen3_4binstruct.yaml
HF_HOME=.cache/ \
WANDB_API_KEY={your W&B key} \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="{your NV username}-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    ++grpo.val_at_start=false \
    ++grpo.num_prompts_per_step=4 \
    ++grpo.max_num_steps=3 \
    ++policy.dtensor_cfg.clear_cache_every_n_steps=1 \
    ++cluster.num_nodes=1 \
    checkpointing.checkpoint_dir=results/$EXP_NAME &
```

Multi node run

```shell
WANDB_API_KEY={your W&B API key} \
SLURM_ACCOUNT=llmservice_modelalignment_sft \
SLURM_PARTITION=batch \
EXP_NAME=20250926/penguin_grpo/qwen3_4binstruct/8nodes/ext_taubench_env_001 \
NUM_ACTOR_NODES=8 \
REPO_LOCATION={your NeMo RL dir} \
    ./temp_penguin_submit.sh \
    --config=examples/penguin/grpo_ext_taubench_env_qwen3_4binstruct.yaml \
    logger.wandb.project="$USER-nemo-gym-rl-integration"
```

### Mini SWE

Prepare data

```shell
config_paths="resources_servers/mini_swe_resource/configs/mini_swe_resource.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/swegym \
    +mode=train_preparation \
    +should_download=true \
    ++mini_swe_main_agent_train.responses_api_agents.mini_swe_agent.cache_dir_template=dummy \
    ++mini_swe_main_agent_validation.responses_api_agents.mini_swe_agent.cache_dir_template=dummy
```

Install singularity on single node

```shell
# Install singularity
apt-get update && apt-get install -y git build-essential gcc
apt install -y wget && \
cd /tmp && \
wget https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb && \
apt install -y ./apptainer_1.3.1_amd64.deb && \
ln -sf /usr/bin/apptainer /usr/bin/singularity
```

Run single node

```shell
pkill -f VllmAsyncGenerationWorker
uv run ray stop --force
uv run python -c "import ray; ray.shutdown()"
EXP_NAME="$(date +%Y%m%d)/penguin_grpo/qwen3_30ba3b_instruct/miniswe_test_001"
CONFIG_PATH=examples/penguin/grpo_miniswe_qwen3_coder_30ba3b_instruct_large_rollout_batch.yaml
HF_HOME=.cache/ \
HF_TOKEN={HF token} \
HF_HUB_OFFLINE=1 \
WANDB_API_KEY={W&B api key} \
RAY_BACKEND_LOG_LEVEL=debug \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="bxyu-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    ++grpo.val_at_start=false \
    ++grpo.num_prompts_per_step=1 \
    ++grpo.max_num_steps=3 \
    ++policy.train_global_batch_size=16 \
    ++policy.max_total_sequence_length=8192 \
    ++policy.dtensor_cfg.clear_cache_every_n_steps=1 \
    ++cluster.num_nodes=1 \
    ++env.penguin.mini_swe_main_agent_train.responses_api_agents.mini_swe_agent.concurrency=16 \
    ++env.penguin.mini_swe_main_agent_val.responses_api_agents.mini_swe_agent.concurrency=16 \
    checkpointing.checkpoint_dir=results/$EXP_NAME &
```

temp\_penguin\_miniswe\_submit.sh

```shell
# ----- PARAMETERS -----
# WANDB_API_KEY, SLURM_ACCOUNT, SLURM_PARTITION, EXP_NAME, NUM_ACTOR_NODES, REPO_LOCATION

# ----- CONSTANTS -----
# CONTAINER_IMAGE_PATH=/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/bxyu/cache/containers/nvcr.io/nvidian/nemo-rl:8762f57-37992537
CONTAINER_IMAGE_PATH=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/nvcr.io/nvidian/nemo-rl:bd2e645-37644239.squashfs

# Setup commands to run on all nodes
read -r -d '' SETUP_COMMAND <<EOF
apt-get update && apt-get install -y git build-essential gcc
apt install -y wget && \
cd /tmp && \
wget https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb && \
apt install -y ./apptainer_1.3.1_amd64.deb && \
ln -sf /usr/bin/apptainer /usr/bin/singularity
cd ${REPO_LOCATION}
EOF

read -r -d '' COMMAND <<EOF
cd ${REPO_LOCATION}

HF_HOME=.cache/ \
HF_HUB_OFFLINE=1 \
WANDB_API_KEY=$WANDB_API_KEY \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    cluster.num_nodes=$NUM_ACTOR_NODES \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    checkpointing.checkpoint_dir=results/$EXP_NAME \
    $@
EOF

echo -e "Running command:\n$COMMAND"

# Not sure why this is necessary, but ray.sub needs to be launched from the NeMo-RL root directory
cd $REPO_LOCATION
COMMAND=$COMMAND \
SETUP_COMMAND=$SETUP_COMMAND \
CONTAINER=$CONTAINER_IMAGE_PATH \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=$NUM_ACTOR_NODES \
    --account=$SLURM_ACCOUNT \
    --partition=$SLURM_PARTITION \
    --time=4:0:0 \
    --job-name=$EXP_NAME \
    --gres=gpu:8 \
    ray.sub
```

Run multinode

```shell
WANDB_API_KEY={W&B API key} \
SLURM_ACCOUNT=llmservice_modelalignment_ppo \
SLURM_PARTITION=batch \
EXP_NAME=20251108/penguin_grpo/qwen3_coder_30ba3b_instruct/32nodes/miniswe_001 \
NUM_ACTOR_NODES=32 \
REPO_LOCATION=/lustre/fsw/portfolios/llmservice/users/bxyu/nemo-gym/RL-dev-2/nemo-rl \
    /lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/bxyu/nemo-gym/temp_penguin_miniswe_submit.sh \
    --config=examples/penguin/grpo_miniswe_qwen3_coder_30ba3b_instruct_large_rollout_batch.yaml \
    logger.wandb.project="$USER-nemo-gym-rl-integration-20250926"
```

## Trajectory collection

### Math

Prepare data

```shell
config_paths="resources_servers/library_judge_math/configs/dapo17k_trajectory_collection.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/dapo17k_trajectory_collection \
    +mode=train_preparation \
    +should_download=true
```

Single node

```shell
pkill -f VllmAsyncGenerationWorker
uv run ray stop --force
uv run python -c "import ray; ray.shutdown()"
EXP_NAME="$(date +%Y%m%d)/penguin_grpo/qwen3_30ba3b_instruct/dapo17k_trajectory_collection_test_001"
CONFIG_PATH=examples/penguin/grpo_dapo17k_bytedtsinghua_qwen3_30ba3b_instruct_trajectory_collection.yaml
HF_HOME=.cache/ \
HF_TOKEN={HF API key} \
HF_HUB_OFFLINE=1 \
WANDB_API_KEY={W&B API key} \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="{Your user name}-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    ++grpo.val_at_start=false \
    ++grpo.num_prompts_per_step=4 \
    ++grpo.max_num_steps=3 \
    ++policy.dtensor_cfg.clear_cache_every_n_steps=1 \
    ++cluster.num_nodes=1 \
    checkpointing.checkpoint_dir=results/$EXP_NAME &
```

Multi node

```shell
WANDB_API_KEY={W&B API key} \
SLURM_ACCOUNT=llmservice_nemotron_nano \
SLURM_PARTITION=batch \
EXP_NAME=20251102/penguin_grpo/qwen3_30ba3b_instruct/32nodes/dapo17k_trajectory_collection_001 \
NUM_ACTOR_NODES=32 \
REPO_LOCATION={your NeMo RL dir} \
    ./temp_penguin_submit.sh \
    --config=examples/penguin/grpo_dapo17k_bytedtsinghua_qwen3_30ba3b_instruct_trajectory_collection.yaml \
    logger.wandb.project="$USER-nemo-gym-rl-integration"
```

## FAQ

### HF Hub rate limits

Not sure what this is caused by, but you can add `HF_HUB_OFFLINE=1` to your submit script. See the script below (the change is bolded)

```shell
# ----- PARAMETERS -----
# WANDB_API_KEY, SLURM_ACCOUNT, SLURM_PARTITION, EXP_NAME, NUM_ACTOR_NODES, REPO_LOCATION

# ----- CONSTANTS -----
CONTAINER_IMAGE_PATH=/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/nvcr.io/nvidian/nemo-rl:bd2e645-37644239.squashfs

read -r -d '' COMMAND <<EOF
cd ${REPO_LOCATION}

HF_HOME=.cache/ \
HF_HUB_OFFLINE=1 \
WANDB_API_KEY=$WANDB_API_KEY \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    cluster.num_nodes=$NUM_ACTOR_NODES \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    checkpointing.checkpoint_dir=results/$EXP_NAME \
    $@
EOF

echo -e "Running command:\n$COMMAND"

# Not sure why this is necessary, but ray.sub needs to be launched from the NeMo-RL root directory
cd $REPO_LOCATION
COMMAND=$COMMAND \
CONTAINER=$CONTAINER_IMAGE_PATH \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=$NUM_ACTOR_NODES \
    --account=$SLURM_ACCOUNT \
    --partition=$SLURM_PARTITION \
    --time=4:0:0 \
    --job-name=$EXP_NAME \
    --gres=gpu:8 \
    ray.sub
```

If you see an error like

```shell
E               OSError: There was a specific connection error when trying to load Qwen/Qwen3-0.6B:
E               429 Client Error: Too Many Requests for url: https://huggingface.co/api/models/Qwen/Qwen3-0.6B/xet-read-token/c1899de289a04d12100db370d81485cdf75e47ca (Request ID: Root=1-68daef4d-65a036f37251659228d72abb;127ae8ae-20aa-4488-a9d0-9bb3d838354c)
E               
E               We had to rate limit your IP (216.228.127.129). To continue using our service, create a HF account or login to your existing account, and make sure you pass a HF_TOKEN if you're using the API.
```

You can also try setting your HF\_TOKEN

### Git is missing a bunch of remote branches

If you see an error profile like this:

```shell
cw-dfw-cs-001-login-01:/lustre/fsw/portfolios/llmservice/users/$USER/nemo-rl-2/3rdparty/Penguin-workspace/Penguin$ git fetch
# Nothing here

cw-dfw-cs-001-login-01:/lustre/fsw/portfolios/llmservice/users/slikhite/nemo-rl-2/3rdparty/Penguin-workspace/Penguin$ git checkout /some/other/branch
error: pathspec '/some/other/branch' did not match any file(s) known to git
```

Please follow the steps outlined in this doc\! [https://stackoverflow.com/questions/11623862/fetch-in-git-doesnt-get-all-branches](https://stackoverflow.com/questions/11623862/fetch-in-git-doesnt-get-all-branches). As of Fri Oct 03, we are unsure what causes this issue.

```shell
# This command will probably output something like `+refs/heads/master:refs/remotes/origin/master`. If it doesn't, you may have a different issue
git config --get remote.origin.fetch

# Set the remote to fetch all branches, not just main
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"

# Check that the config is correctly updated. This command should output `+refs/heads/*:refs/remotes/origin/*`.
git config --get remote.origin.fetch

# This should fetch all branches. You should see a bunch of downloading
git fetch
```

### Import NeMo RL container

Get access to the nvidian org for NGC to access the NeMo RL private registry [https://nvidia.enterprise.slack.com/archives/C7VGNG1V3](https://nvidia.enterprise.slack.com/archives/C7VGNG1V3)   
Then visit [https://registry.ngc.nvidia.com/orgs/nvidian/containers/nemo-rl/tags](https://registry.ngc.nvidia.com/orgs/nvidian/containers/nemo-rl/tags) for the list of available containers.

```shell
srun -A llmservice_modelalignment_sft -p interactive \
    --gres=gpu:8 \
    --nodes=1 --ntasks=1 --time 04:00:00 \
    --pty /bin/bash

# One time setup
echo 'machine nvcr.io login $oauthtoken password {NGC API key}' >> ~/.config/enroot/.credentials
mkdir -p cache/containers/nvcr.io/nvidian

# Import container
CONTAINER_IMAGE_PATH="./cache/containers/nvcr.io/nvidian/nemo-rl:25.11.rc0.sqsh"
enroot import -o "$CONTAINER_IMAGE_PATH" 'docker://nvcr.io/nvidian/nemo-rl:25.11.rc0'
```

### self.lr\_decay\_steps \> 0

[https://wandb.ai/nvidia/bxyu-nemo-gym-rl-integration-20250926/runs/x5dag3jj/logs?nw=nwuserbxyu](https://wandb.ai/nvidia/bxyu-nemo-gym-rl-integration-20250926/runs/x5dag3jj/logs?nw=nwuserbxyu)  
Your batch size is too big for your data


## What You've Learned
You now understand how to train a model with NeMo Gym and NeMo RL.

- **Core concepts**: What rollouts are and why they're fundamental to NeMo Gym
- **Generation workflow**: From input tasks to complete interaction records
- **Practical skills**: Using `ng_collect_rollouts` with different strategies
- **Data analysis**: Understanding rollout structure and analyzing results
- **Best practices**: Efficient and reliable rollout generation

TODO: data preparation and basics of RL training framework integration.

**Next steps**: 
- **[Next: Offline Training with Rollouts (SFT/DPO)](07-sft-dpo-rollout-collection.md)**
