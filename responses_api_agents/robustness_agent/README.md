# Description

Use this agent when you need the same functionality as `simple_agent` but want to train with prompt rewriting, tool name rewriting, or argument name rewriting. This should help with prompt sensitivity, tool overfitting, and model generalization, but is less controllable than pre-generated diverse prompts and tool names. The value is that it can be easily applied to any enviornment using `simple_agent` with just a config change. 

Please see the config to configure the agent, for example: 


```
      rewrite_prompts: true
      rewrite_tool_names: false
      rewrite_arg_names: false
      rewrite_prob: 0.2
```


# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
