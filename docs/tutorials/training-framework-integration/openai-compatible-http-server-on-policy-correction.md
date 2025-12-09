(openai-compatible-http-server-on-policy-correction)=

# OpenAI-compatible HTTP server On-Policy correction
In addition to OpenAI-compatible HTTP server functionality, a fundamental issue exists in {term}`multi-step <Multi-step>` and {term}`multi-turn <Multi-turn>` scenarios.

## Preliminaries

### Preliminary 1: Life cycle of an OpenAI HTTP request
The life cycle of a single OpenAI HTTP request is as follows. Each step produces a single output.
1. "Request": A JSON payload representing a rollout is sent to the HTTP server endpoint.
   1. The schema here is typically in Responses input items or Chat Completions messages. For the rest of the sequence, we will assume Responses input items.
2. "Input prompt": The Responses input items are "chat templated" i.e. converted from objects into a single string.
3. "Prompt token IDs": The input prompt is "tokenized" i.e. converted from a string into a sequence of model-understandable token IDs.
4. "Generation token IDs": The prompt token IDs are sent to the model and the model generations a new sequence of token IDs.
5. "Generation": The generation token IDs are "de-tokenized" into a string.
6. "Response": The generation is "parsed" into a sequence of Responses output items and returned.

We refer to the above life cycle steps 1-6 as LF1 - LF6.

### Preliminary 2: Life cycle of a rollout
A single multi-step or multi-turn rollout will make multiple requests in sequence to model endpoint.
For example, a multi-step multi-turn rollout with 2 turns may look like:
1. [First turn] User message
2. Assistant Reasoning message (typically shortened to just Reasoning)
3. Assistant Chat
4. Assistant Tool Call
5. Tool response
6. Reasoning
7. Chat
8. Tool Call
9. Tool
10. [First turn, third step] Chat
11. [Second turn] User message
12. ...

For the purposes of this doc, we will abbreviate rollouts like above by message type: U R C TC T R C TC T C U. U and T messages are obtained entirely independently of the model endpoint. R, C, and TC messages are obtained from the model endpoint.

Today, most model endpoints will return [R C TC] messages in a single response. So in effect, the rollout can be viewed as U [R C TC] T [R C TC] T [C] U, where items in brackets [] indicate a single model call (MC).

### Preliminary 3: Train and generation time log probability mismatch
Policy optimization algorithms typically have rollout and train phases within a single step. Policy optimization algorithms calculate and backpropagate through a loss calculated using log probabilities (logprobs). This leads to situations where the rollout logprobs and token selections may differ from the actual logprobs calculated at train time, leading to off-policy training. This phenomenon is typically described as "train-generation mismatch".

Larger mismatches lead to further off-policyness during training. While policy optimization algorithms can tolerate a small amount of off-policyness, too much typically leads to training run crashes.


## Problems

We observe several problems in practice when using an OpenAI-compatible HTTP server during training that cause train and generation time log probability mismatch, leading to significant off-policyness.


### Problem 1: Re-tokenization
Observe that going from LF5 of the previous model call to LF3 of the current model call (i.e. across model calls) implies that we lose information that was once held in token IDs.

For example, in the previous model call, the model may have produced token IDs 1 and 2 which were detokenized into `_Skinny` in LF5. Then in step LF3, we could re-tokenize `_Skinny` into token ID 3.

At the previous model call generation time, the logprobs for the tokens following those for `_Skinny` will be calculated using token ID 1 and token ID 2. Then, when we go to train, the logprobs for the tokens following `_Skinny` will be calculated using token ID 3, leading to a logprob mismatch.

We've seen the following situations:
1. Merging upon re-tokenization: model produces token IDs 1 and 2 which is later re-tokenized to a single token ID 3 e.g. "_Ski" + "nny" -> "_Skinny".
2. Different re-tokenization: model produces token IDs 1 and 2 which is later re-tokenized to two different token IDs 3 and 4 e.g. "_Ski" + "nny" -> "_Skin" + "ny".


### Problem 2: Re-chat templating
Observe that going from LF6 of the previous model call to LF2 of the current model call implies that we lose information that was once held in the generation.

For example, in the previous model call at LF6, the model may have produced token IDs that were de-tokenized into `<tool_call><name>get_weather</name><parameters>{"city": "SF"}</parameters></tool_call>`. This is then converted to an OpenAI tool call i.e. `{"type": "function", "function": "get_weather", "arguments": "{\"city\": \"SF\"}"}`.

Then in LF2, we will re-chat-template this OpenAI tool call into a string. But the way the chat template represents this tool call may differ from what the model actually produced e.g. `<tool_call>\n<name>\nget_weather\n</name>\n<parameters>\n{"city": "SF"}\n</parameters>\n</tool_call>\n` adding newlines in between each opening and closing XML tag.

This isn't just because the chat template is esoteric or incorrect. Due to the stochastic nature of models today, we have no guarantee that the model will produce the same exact format as the deterministic process of the chat template.


### Problem 3: Non-monotonically increasing history
Sometimes developers will intentionally change the history of a rollout.
1. Agentic coding harnesses will sometimes summarize or truncate entirely prior history as a rollout gets longer and longer over steps.
2. Various model chat templates will remove reasoning from the input prompt across turns.

This directly changes the prompt token IDs the model sees at the current model call and differs from the final prompt token IDs used for training.


## Current solution
We solve the problems above using two components:
1. For Problems 1 and 2, we will perform the following on-policy token IDs fix in the vLLM OpenAI HTTP server https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/nemo_rl/models/generation/vllm/vllm_worker_async.py#L40
2. For Problem 3, we will disable reasoning truncation across turns using the chat template.
   1. It's an open research question on how to train on these types of scenarios.

The on-policy token IDs fix in the vLLM OpenAI HTTP server assumes the following prerequisites:
1. `model_prefix_token_ids`: The ground truth prompt token IDs concatenated with the generation token IDs from the previous model call.
2. `template_prefix_token_ids`: The re-templated and re-tokenized token IDs up to and not including the final assistant message.
3. `template_token_ids`: The re-templated and re-tokenized token IDs for the entire rollouts.
   1. We assume that `template_prefix_token_ids` is a strict prefix of `template_token_ids` (specifically we need to circumvent Problem 3).

Our fix will find the position of the right EOS token ID in the `template_token_ids` and splice in the `model_prefix_token_ids`.

For example:
1. Let's assume the current request in LF1 contains the following rollout structure: `U A T A U A T`
2. The variables here would be:
   1. `model_prefix_token_ids`: Ground truth token IDs that correspond to `U A T A U A`.
   2. `template_prefix_token_ids`: Re-templated and re-tokenized token IDs that correspond to `U A T A U A`.
   3. `template_token_ids`: Re-templated and re-tokenized token IDs that correspond to `U A T A U A T`.
3. We will use `template_prefix_token_ids` to find the position of the EOS token ID corresponding to the content represented in `model_prefix_token_ids`.
4. Splice the prefix of `template_token_ids` and replace it with `model_prefix_token_ids`.
