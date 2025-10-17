# TALES Resource Server

Integrates: [TALES](https://github.com/microsoft/tale-suite)

Specifically, this uses the tt_split branch which provides both a set of tasks for training and evaluation.

Partially based on code at [draft textworld implementation](https://github.com/NVIDIA-NeMo/Gym/tree/cmunley1/textworld/resources_servers/textworld) and [workbench](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/workbench)

You need Java to run Scienceworld. If you don't have it, you'll get an error message but should otherwise be uneffected if you are using other environments.

This can be installed with 
`sudo apt-get update && apt-get install default-jre default-jdk`


1. Environment: This is an implementation of TALES: a multi-framework agentic environment that evaluates an agent's ability to reason through and progress through open-ended, situated, text-environments. TALES consists of 5 text-adventure game frameworks with a total of 122 tasks (games) with the rough order difficulty of textworld, textworld_express, alfworld, scienceworld, and jericho.

2. Domain: Text-Game Environments
3. Source of prompts: 
- The prompt at each turn consists of the system prompt as well as the observation-action history of the agent up to that step.
- Our examples use the same system prompt used in TALES, however, this can be adjusted to user preference.
4. Example prompt:
- Please see any of the *_clean.jsonl examples under data/gpt4o_examples. 
5. Verifier: Rewards are provided via the underlying gymnasium environment. 
6. Legal Approval Status: N/A


# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
