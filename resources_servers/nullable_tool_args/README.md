# Description

Checks that a model fills required nullable tool arguments (`anyOf [string, null]` or `type: [string, null]`) with JSON `null` rather than the string `"None"`. Models trained on chat templates that render null tool arguments with Python `str()` tend to write `None`, which tool-call parsers keep as a string.

Each task asks for one tool call where the arguments named in `verifier_metadata.null_args` should be null. Reward is 1.0 when the first function call sets all of them to null, else 0.0. `verifier_metadata.position` is `first` (no earlier tool calls) or `mid` (an earlier correct call with null arguments is in the history), and `first/null_rate` and `mid/null_rate` are reported separately. Tool endpoints only acknowledge the call.

```bash
gym env start --resources-server nullable_tool_args --model-type vllm_model
```

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
