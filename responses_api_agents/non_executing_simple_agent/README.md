# Non-Executing Simple Agent

This agent forwards a single `/v1/responses` request to the configured model server and passes the model response directly to the configured resource server verifier.

It is intended for tasks where a tool call is the final answer. It does not execute tool calls, parse tool-call arguments, or append tool outputs.
