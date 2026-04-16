# Description
This is a responses API agent that is intended to be used with malformed_think_and_tool_calls_verifier, which is a proxy for checking if the model is able to think properly and format tool calls properly.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0

# Test
```
ng_test +entrypoint=responses_api_agents/malformed_think_and_tool_calls_verifier_agent
```