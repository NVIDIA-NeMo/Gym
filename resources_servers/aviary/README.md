> Keywords: Tool Use, Multi-step Reasoning, Environment Interaction, Scientific Tasks

This resources server adapts [Aviary environments](https://github.com/Future-House/aviary) into the NeMo Gym resources-server interface, so NeMo Gym agents can interact with Aviary `Environment`s. This allows one to implement tool and environment logic in Aviary, and deploy the environment for inference or training with Gym.

### Implemented servers in this folder

- **GSM8K**: `gsm8k_app.py`
  - Meant primarily as an example, this implements [GSM8k](https://arxiv.org/abs/2110.14168) as a set of environments equipped with a calculator tool.
- **HotPotQA**: `hotpotqa_app.py`
  - The HotPotQA environment asks agents to perform multi-hop question answering on the [HotPotQA dataset](https://aclanthology.org/D18-1259/)
- **BixBench**: `notebook_app.py`
  - Implements the [BixBench dataset](https://arxiv.org/abs/2503.00096) as a set of environments that allow execution of a Jupyter notebook.
  - Also serves as an example for how to implement notebook-backed environments for other scientific computational tasks.
- **BixBench-Hypothesis**: `hypotest_app.py (bundled), client_app.py (remote)`
  - Implements the [BixBench-Hypothesis dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-bixbench_hypothesis) with a REPL-style code interface and testing hypothesis testing capabilities.
- **Client/proxy to a remote Aviary dataset server**: `client_app.py`
  - A generic interface to an Aviary `TaskDatasetServer`. Can be used to interact with any Aviary environments being served remotely.


# Example usage

Run the GSM8K Aviary resources server together with a model config:

```bash
config_paths="resources_servers/aviary/configs/gsm8k_aviary.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[$config_paths]"
```

Then collect rollouts:

```bash
ng_collect_rollouts \
    +agent_name=gsm8k_aviary_agent +input_jsonl_fpath=resources_servers/aviary/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/aviary/data/example_rollouts.jsonl
```

# BixBench-Hypothesis (BBH)
BixBench-Hypothesis is a dataset proposed by Edison Scientific to measure LLM capabilities for testing hypotheses in bioinformatics contexts. Edison Scientific and NVIDIA have also collaborated to release BBH-Train, an RL training dataset meant to improve model capabilities on bioinformatics-related data analysis.

There are two methods to running BBH with NeMo-Gym: the remote approach and the bundled approach. The remote approach hosts the environment sandboxes as an external service that NeMo-Gym can communicate with, serving as a modular method for running train/inference jobs. The bundled approach colocates the environment sandboxes on the Gym/RL nodes, serving as an efficient and fully packaged method for running train/inference jobs.

## Remote Approach:
To run the remote approach, first launch the dataset server implemented in the [hypotest](https://github.com/EdisonScientific/hypotest) repository. Documentation on how to run the dataset server can be found [here](https://github.com/EdisonScientific/hypotest/blob/main/README.md).

Then, prepare your Gym data with the task_idx values of the problems you would like to train/evaluate on. An example dataset is provided for reference in [data/bbh_train_example.jsonl](data/bbh_train_example.jsonl).

Once the dataset server is running and is accessible at a specific URL, update your config based on [configs/bbh_remote.yaml](configs/bbh_remote.yaml) with the server URL and api key, and launch NeMo-Gym as follows:

```bash
config_paths="resources_servers/aviary/configs/bbh_remote.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[$config_paths]"
```

Then collect rollouts on your data as follows (updating the input_jsonl_fpath to your Gym data if needed):

```bash
ng_collect_rollouts \
    +agent_name=bbh_aviary_agent +input_jsonl_fpath=resources_servers/aviary/data/bbh_train_example.jsonl \
    +output_jsonl_fpath=resources_servers/aviary/data/example_bbh_rollouts.jsonl
```

To run training with NeMo-RL, set the following fields in your NeMo-RL container (where train_data.jsonl and validation_data.jsonl are set to your train/val Gym data respectively, and bbh_remote.yaml is updated with your dataset server URL/api-key):
```yaml
data:
  train_jsonl_fpath: 3rdparty/Gym-workspace/Gym/resources_servers/aviary/data/train_data.jsonl
  validation_jsonl_fpath: 3rdparty/Gym-workspace/Gym/resources_servers/aviary/data/validation_data.jsonl
  shuffle: False
  num_workers: 1

env:
  should_use_nemo_gym: true
  nemo_gym:  # This is passed into NeMo-Gym as the initial_global_config_dict
    is_trajectory_collection: false  # Set this to true to enable trajectory collection (no training). You may also want to increase `policy.generation.vllm_cfg.gpu_memory_utilization`
    config_paths:
    - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml  # Required! And it must be *for_training
    - resources_servers/aviary/configs/bbh_remote.yaml 
```

Note that task_idx values in your Gym data must align with the data in [hypotest](https://github.com/EdisonScientific/hypotest); this means that both your train and val set problem data must be provided to [hypotest](https://github.com/EdisonScientific/hypotest)'s dataset server.

## Bundled Approach:
To run the bundled approach, first update the config in [configs/bbh_bundled.yaml](configs/bbh_bundled.yaml) with your desired configuration. The config fields in `dataset` closely match the config fields in [hypotest](https://github.com/EdisonScientific/hypotest). You'll also have to set [container_sqsh_path], which will be a path to a .sqsh file built from the [hypotest](https://github.com/EdisonScientific/hypotest) Docker container using [enroot](https://github.com/NVIDIA/enroot).

You will also need to add your BBH problem data to [data/](data/), and update `capsule_dir` and `work_dir` with paths to your BBH capsule data and working directory. Note that the working directory must be set to a directory accessible to all nodes if you are running with multi-node jobs, in order for the environment to properly parallelize sandboxes across all available nodes (e.g. the working dir could be made be available on a network-filesystem like lustre).

Once you have your environment properly configured, you'll need to make sure [enroot](https://github.com/NVIDIA/enroot) is installed into your Gym environment. In order to do this, make sure to run the following snippet before bringing up NeMo-Gym:
```bash
cd /tmp &&
apt-get update &&
arch=$(dpkg --print-architecture) &&
curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v4.1.1/enroot_4.1.1-1_\${arch}.deb &&
curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v4.1.1/enroot+caps_4.1.1-1_\${arch}.deb &&
apt install -y ./*.deb &&
apt-get install -y squashfuse &&
cd /path/to/gym/directory
```
And then bring up NeMo-Gym:
```bash
config_paths="resources_servers/aviary/configs/bbh_bundled.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[$config_paths]"
```
```bash
ng_collect_rollouts \
    +agent_name=bbh_aviary_agent +input_jsonl_fpath=resources_servers/aviary/data/bbh_train_example.jsonl \
    +output_jsonl_fpath=resources_servers/aviary/data/example_bbh_rollouts.jsonl
```

If you are running training with NeMo-RL and NeMo-Gym, add the following modification to your `ray.sub` file in NeMo-RL to support adding a setup command:
```diff
diff --git a/ray.sub b/ray.sub
index 9b4feb11..f765a609 100644
--- a/ray.sub
+++ b/ray.sub
@@ -50,6 +50,7 @@ maybe_gres_arg() {
 CONTAINER=$CONTAINER
 MOUNTS=$MOUNTS
 COMMAND=${COMMAND:-}  # This is a script relative to the SLURM_SUBMIT_DIR. If left empty, it will leave the cluster idle after it's brought up.
+SETUP_COMMAND=${SETUP_COMMAND:-}  # Setup commands to run on all nodes before starting Ray
 ########################################################
 # Ports for all nodes (should be odd numbers since we place head/worker[0] on the same node) so all workers get the odd ports, but the head will get +1 the ports
 NODE_MANAGER_PORT=${NODE_MANAGER_PORT:-53001}
@@ -293,6 +294,7 @@ chmod +x /launch-head.sh
 
 count=0
 while [[ \$count -lt $num_retries ]]; do
+  $SETUP_COMMAND
   bash /launch-head.sh
   count=\$((count+1))
   echo "Head node failed \$count/$num_retries times, restarting in 5 seconds..."
@@ -305,6 +307,7 @@ EOF
 srun $COMMON_SRUN_ARGS --container-name=ray-head --nodes=1 --ntasks=1 --cpus-per-task=$CPUS_PER_WORKER -w "$head_node" -o $LOG_DIR/ray-head.log bash -x -c "$head_cmd" &
 SRUN_PIDS["ray-head"]=$!
 
+sleep 100s
 NUM_ACTORS=$((GPUS_PER_NODE * SLURM_JOB_NUM_NODES))
 
 # Start Ray worker nodes
@@ -392,6 +395,7 @@ EOFINNER
 
 count=0
 while [[ \$count -lt $num_retries ]]; do
+  $SETUP_COMMAND
   bash /launch-worker.sh
   count=\$((count+1))
   echo "Worker failed \$count/$num_retries times, restarting in 5 seconds..."
```
Then set up your NeMo-RL config:
```yaml
data:
  train_jsonl_fpath: 3rdparty/Gym-workspace/Gym/resources_servers/aviary/data/train_data.jsonl
  validation_jsonl_fpath: 3rdparty/Gym-workspace/Gym/resources_servers/aviary/data/validation_data.jsonl
  shuffle: False
  num_workers: 1

env:
  should_use_nemo_gym: true
  nemo_gym:  # This is passed into NeMo-Gym as the initial_global_config_dict
    is_trajectory_collection: false  # Set this to true to enable trajectory collection (no training). You may also want to increase `policy.generation.vllm_cfg.gpu_memory_utilization`
    config_paths:
    - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml  # Required! And it must be *for_training
    - resources_servers/aviary/configs/bbh_bundled.yaml 
```
Following this, you can export `SETUP_COMMAND` to a snippet installing enroot prior to launching your `ray.sub` command:
```bash
read -r -d '' SETUP_COMMAND <<EOF
cd /tmp &&
apt-get update &&
arch=$(dpkg --print-architecture) &&
curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v4.1.1/enroot_4.1.1-1_\${arch}.deb &&
curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v4.1.1/enroot+caps_4.1.1-1_\${arch}.deb &&
apt install -y ./*.deb &&
apt-get install -y squashfuse &&
cd $PWD
EOF
export SETUP_COMMAND

COMMAND="[insert nemo rl launch command with gym config]" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=1:0:0 \
    --gres=gpu:8 \
    ray.sub
```

# Licensing information
Code: Apache 2.0

Data: MIT (GSM8k),  Apache 2.0 (BixBench, BixBench-Hypothesis)

Dependencies
- nemo_gym: Apache 2.0
- aviary:  Apache 2.0
- hypotest: Apache 2.0