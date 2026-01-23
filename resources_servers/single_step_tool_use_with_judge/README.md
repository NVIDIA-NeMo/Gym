# Description
This is a resource server that is to be used to verify a single action taken by an agent that can either call a tool or send a chat message to the user as the next step in a trajectory.  An LLM is used as a judge to evaluate the action of the agent and assign a reward to it.

Data links: ?

# Example usage

## Running servers
The following command can be used to run this resource server, along with the tool simulation agent and an OpenAI model:
```bash
config_paths="resources_servers/single_step_tool_use_with_judge/configs/single_step_tool_use_with_judge.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]" \
    +single_step_tool_use_with_judge_resources_server.resources_servers.single_step_tool_use_with_judge.model_step_verifier_config.evaluation_model_server.name=policy_model
```

In this example, the policy model, which is serving as the agent, is also used as the model that evaluates the action of the agent.

Then, rollouts can be collected using a command such as the following:
```bash
ng_collect_rollouts \
    +agent_name=single_step_tool_use_with_judge_agent \
    +input_jsonl_fpath=resources_servers/single_step_tool_use_with_judge/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/single_step_tool_use_with_judge/data/example_rollouts.jsonl
```

# Licensing information
Code: Apache 2.0<br>
Data: ?

Dependencies
- nemo_gym: Apache 2.0
