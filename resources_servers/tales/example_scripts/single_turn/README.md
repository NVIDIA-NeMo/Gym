# TALES Example Single-Turn Script Set Up

Scripts for generating example trajectories from NeMo Integration of [TALES](https://github.com/microsoft/tale-suite).

You need Java to run Scienceworld. If you don't have it, you'll get an error message but should otherwise be uneffected if you are using other environments.
This can be installed with 
```bash
sudo apt-get update && apt-get install default-jre default-jdk
```

## Limitations
- Generating examples for Scienceworld can take particularly long due to the number of actions required to complete some tasks in this framework.

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

4. For step 2, Simple Correctness Check, in the CONTRIBUTING.md, the following script generates 5 gpt-4o rollouts and dumps them in data under `gpt4o_single_turn_examples`. A example.jsonl is generated and placed directly under the data folder for the PR unit tests.
```bash
python resources_servers/tales/example_scripts/single_turn/generate_single_turn_gpt_rollouts.py 
```

5. For step 4, Reward Profiling, in the CONTRIBUTING.md, the following script generates 500+ prompt-response pairs for the specified model. As TALES is inherently multi-turn, not every step on a correct trajectory will return a reward. We do the following to emulate the reward distribution for single-turn domains.

- We extract the walkthrough actions, a_0, a_1, ..., a_k, from the underlying gym environment where a_k represents the action in the walkthrough at step k. 
- We feed these actions to the environment to obtain trajectory (obs_0, a_0, r_0), (obs_1, a_1, r_1), ..., (obs_k, a_k, r_k).
- For each turn when reward, r_n, is non-zero, we provide the LLM the game history (obs_0, a_0), (obs_1, a_1), ..., (obs_n-1, a_n-1) and obs_n while asking the LLM to predict action a_n. We do this 16 times.
- For each unique predicted action a_pred, we fast-forward the gym environment to step n-1 and pass a_pred into the environment to see if it is accepted.

We do this across all of the tasks in textworld, textworld_express, alfworld, and scienceworld for walkthroughs of length less than 5, resulting in a total of ~600 prompts for both Qwen3-30B-A3B and Qwen3-235B-A22B. For Qwen3-30B-A3B, we prepend /no_think to the beginning of the user input to prevent thinking. If you are running this yourself, generating the prompts and responses for Scienceworld will take some time, due to how long the prompts can get.

> Remember to change the provided port (if needed) and the output folder. Other than that, you should be able to just run the same command.

```bash
python resources_servers/tales/example_scripts/single_turn/generate_single_turn_rollouts.py 
```