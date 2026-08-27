# Reward Profiling How-To

## Index
1. (Optional) Create a container
2. Create a manifest
3. Run the reward profiling job
    a. Sharding / Unsharding data
    b. Starting, resuming and monitoring a profiling job
4. Postprocess reward profiling outputs
    a. Collating finished / unfinished data
    b. Running ng_reward_profile
    c. Re-creating profiled data to input shapes with reward profiled information.


## 01 - (Optional) Create a container
The reward profiling container pre-installs all resources_servers and responses_api_agents needed for reward profiling.
It follows the same flow from the [Super-v3.5 readme](https://github.com/NVIDIA-NeMo/Gym/blob/main/benchmarks/nemotron_3.5_super/README.md), with one change:
1. New container config: `benchmarks/nemotron_3.5_super/reward_profiling/configs/container_config.yaml`
blah blah blah how to create an eval container using the container config.yaml

## 02 - Create a Manifest
The manifest is the highest-level config of what environments are being profiled, and parameterizes any judge, sandbox, or config overrides needed.
1. nickname: name of the reward profiling jobs
2. defaults: set any reward profiling defaults.
    - can set: num_repeats, {add more that can be parameterized}
3. extra_configs: any other configs to be loaded (idk how this works explain)
4. config_overlay: any overrides that are applied to gym_env_start (or whichever it is, explain)
    - this is how we route different judge models to one gym_env_start server
5. entries: the environments to be profiled.
    - label (required): nickname of profiled env
    - agent (required ? optional): agent_ref of the data
    - configs (required): required gym configs to run the resources_server/agent whatever expalin
    - data (required): jsonl path to the data with labelled agent_ref
    - owner (optional): owner of environment

## 03 - Run the reward profiling job
idk explain concisely

### 03a - Sharding / Unsharding data
idk explain concisely
