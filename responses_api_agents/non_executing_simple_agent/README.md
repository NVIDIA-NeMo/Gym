# Non-Executing Simple Agent

> **Deprecated.** Use `simple_agent` with `execute_tools: false` under a `single_agent_turn_legacy`
> Environment Server instead. It makes the same single model call and passes its tool calls to the verifier
> unexecuted, and as an agent session it resumes inside its turn after a crash. The shipped configs that used
> this agent have moved; this agent will be removed in a later release.

This agent forwards a single `/v1/responses` request to the configured model server and passes the model response directly to the configured resource server verifier.

It is intended for tasks where a tool call is the final answer. It does not execute tool calls, parse tool-call arguments, or append tool outputs.
