# Description
This is a resource server that is to be used to verify a single action taken by an agent that can either call a tool or send a chat message to the user as the next step in a trajectory.  For each verification request, there is an expected action that is either a tool call or a chat message.  An expected tool call is compared with a tool call issued by the agent by programmatically comparing the arguments in the tool calls.  If the expected action is a chat message, then the agent receives a positive reward if it sends a chat message, and a negative reward if it calls a tool instead.

Data links: ?

# Example usage

## Running servers
The following command can be used to run this resource server, along with the tool simulation agent and an OpenAI model:
```bash
config_paths="resources_servers/single_step_tool_use_with_argument_comparison/configs/single_step_tool_use_with_argument_comparison.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

Then, rollouts can be collected using a command such as the following:
```bash
ng_collect_rollouts \
    +agent_name=single_step_tool_use_with_argument_comparison_agent \
    +input_jsonl_fpath=resources_servers/single_step_tool_use_with_argument_comparison/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/single_step_tool_use_with_argument_comparison/data/example_rollouts.jsonl
```

# Licensing information
Code: Apache 2.0<br>
Data: ?

Dependencies
- nemo_gym: Apache 2.0
