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

## CONTRIBUTING.md Notes:
All of the .jsonl files that are in the top level of the data folder are from the single_turn examples. We place them here to pass the checks on the PR.

## Key App Functions:
*/seed_session*:
This is largely used to initialize the environment as well as switch tasks (CookingWorld1, CookingWorld2, etc) or frameworks (Textworld, Textworld_Express, etc).
This function also automatically calls 'reset' and returns the corresponding observation and information. If you want to switch the environment, you can call /seed_session. If you want to just reset the current environment, call /reset.
Input example:
```
{
  "framework": "alfworld",
  "split": "train",
  "task_no": 0,
  "seed": 123
}
```
All arguments are optional and will default to the values in the config if not used.

Response example:
```
{
  "observation": "You are in a kitchen...",
  "score": 0.0,
  "done": false,
  "info": { "admissible_commands": ["look", "inventory"] },
  "session_id": "c8a1f3e2-...",
  "available_tasks": 54,
  "admissible_commands": ["look", "inventory"]
}
```

*/execute_command*:
Sends the command to the environment and returns the output. For now, we need to include the `session_id` due to some issues with the wrong server being queried otherwise. If `session_id` is not included, then we default to the last used environment by the server.
Input example:
```
{
  "session_id": "c8a1f3e2-...",
  "command": "look"
}
```
Response example:
```
{
  "observation": "You see a fridge and a table...",
  "score": 0.0,
  "done": false,
  "info": { "...": "..." },
  "admissible_commands": ["open fridge", "open table drawer"]
}
```

*/verify*:
Since the environment itself acts as a verifier, we basically treat the `verify` function as a step command, the same as `execute_command`. For now, we're keeping them seperate but may change them in the future.

Input example:
```
{
  "responses_create_params": {The input to the model...},
  "response": {"content":["text": "do something..."]}
}
```
Response example:
```
{
  "observation": "Feedback from doing something...",
  "score": 0.0,
  "done": false,
  "info": { "...": "..." },
  "admissible_commands": ["open fridge", "open table drawer"],
  "responses_create_params": {The input to the model...},
  "response": {"content":["text": "do something..."]}
}
```


*/reset*:
Resets the environment.

Input example:
```
{
  "session_id": "c8a1f3e2-..."
}
```
Response example:
```
{
  "observation": "You are in a kitchen...",
  "score": 0.0,
  "done": false,
  "info": { "...": "..." },
  "admissible_commands": ["look", "inventory"]
}
```
