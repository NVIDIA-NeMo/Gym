# TALES Resource Server

Integrates: [TALES](https://github.com/microsoft/tale-suite)

Specifically, this uses the tt_split branch which provides both a set of tasks for training and evaluation.

Partially based on code at [draft textworld implementation](https://github.com/NVIDIA-NeMo/Gym/tree/cmunley1/textworld/resources_servers/textworld) and [workbench](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/workbench)

You need Java to run Scienceworld. If you don't have it, you'll get an error message but should otherwise be uneffected if you are using other environments.

This can be installed with 
`sudo apt-get update && apt-get install default-jre default-jdk`

## Multi-Turn vs. Single-Turn
TALES is natively a multi-turn environment for evaluating the reasoning capabilities of LLMs, where observations are provided as `user` inputs while the agent's past actions are represented as `assistant` inputs.

Currently, we are encountering some issues regarding our multi-turn implementation with regard to single-turn constraints within the NeMo Gym. The draft code and some examples can be found under their respective folders in data and example_scripts.

For compatability with NeMo Gym, we provide an alternative single-turn formulation where the entire observation-action history is condensed into the content of the first 'user' message and the LLM is queried for the next action.

Please see the README.md under the single_turn and multi_turn folders of example_scripts for more information and code samples.

## Verifiers and Walkthroughts
All environments within TALES have a reward (or score) provided by the underlying gynmasium implementation. Because of this, the environment itself acts as a verifier.

Walkthroughs for each environment are also provided. (See generate_single_turn_rollouts.py under example_scripts/single_turn for an example of how to access them.)

Note, the action sequence in the walkthroughs are often not the only unique sequence of actions that can complete the task. A number of environments in TALES also use nearest-neighbour parsers. This means that, for example, the actions `take lantern`, `get lantern`, and `pick up lantern`, may all be interpreted to be declaring the same intent. For this reason, we recommend that the action is sent to the environment itself for feedback rather than attempting string-matching with the walkthrough.

<!-- ## CONTRIBUTING.md Quality Control Information:

1. Environment: This is an implementation of TALES: a multi-framework agentic environment that evaluates an agent's ability to reason through and progress through open-ended, situated, text-environments. TALES consists of 5 text-adventure game frameworks with a total of 122 tasks (games) with the rough order difficulty of textworld, textworld_express, alfworld, scienceworld, and jericho.

2. Domain: Text-Game Environments
3. Source of prompts: 
- The prompt at each turn consists of the system prompt as well as the observation-action history of the agent up to that step.
- Our examples use the same system prompt used in TALES, however, this can be adjusted to user preference.
4. Example prompt: We provide both single-turn and multi-turn examples. Plea
- Please see any of the *_clean.jsonl examples under data/gpt4o_examples. 
5. Verifier: Rewards are provided via the underlying gymnasium environment. Ground-truth walkthroughs are provided for each environment, however these walkthroughs do not encompass the full range of action sequences that can successfully complete the corresponding task.
6. Legal Approval Status: N/A


# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
? -->
