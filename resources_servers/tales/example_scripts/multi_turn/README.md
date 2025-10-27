# TALES Example Multi-Turn Script Set Up

THIS IS A WORK-IN-PROGRESS DRAFT SECTION. PLEASE MAKE SURE TO NOTE THE LIMITATIONS BELOW WHEN REFERENCING CODE.

Scripts for generating example trajectories from NeMo Integration of [TALES](https://github.com/microsoft/tale-suite).

You need Java to run Scienceworld. If you don't have it, you'll get an error message but should otherwise be uneffected if you are using other environments.
This can be installed with 
```bash
sudo apt-get update && apt-get install default-jre default-jdk
```

## Limitations
- We use a standard vllm server to generate the rollouts for now due to issues getting the NeMo vllm integration working.

## Commands to generate examples

1. Make sure VLLM is installed and follow the steps to run the simple_weather example in the main repository. 

2. Run the following command in another terminal to start up the VLLM server:
```bash
vllm serve Qwen/Qwen3-30B-A3B --enable-expert-parallel --host 0.0.0.0 --port 8000
```

3. Start the TALES Gym Server:
```bash
source .venv/bin/activate

config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/tales/configs/tales.yaml"

ng_run "+config_paths=[$config_paths]" 
```

4. For step 2 in the CONTRIBUTING.md, the following script generates 5 gpt-4o rollouts and dumps them in data under 'gpt_4o_rollouts'.
```bash
python resources_servers/tales/example_scripts/multi_turn/generate_gpt_rollouts.py
```
*TALES uses 100 steps over 5 seeds for reliability. Only 25 steps are used for the trajectories here to avoid excessively long rollouts.

5. For step 4 in the CONTRIBUTING.md, the following script generates 500 rollouts for the specified model. These 500 rollouts are generated with the following splits:

- Ten test tasks from each framework is taken: (Textworld, Textworld Express, Alfworld, Scienceworld, Jericho)
- For each task, we generate 10 rollouts with different seeds

This results in 50 (tasks) x 10 (seeds) 500 total rollouts. As previously, we cap generations at 25 steps to avoid excessively long rollouts.
```bash
python resources_servers/tales/example_scripts/multi_turn/generate_qwen_rollouts.py
```