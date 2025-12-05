(openai-compatible-http-server-on-policy-correction)=

# OpenAI-compatible HTTP server On-Policy correction
In addition to OpenAI-compatible HTTP server functionality, a fundamental issue exists in {term}`multi-step <Multi-step>` and {term}`multi-turn <Multi-turn>` scenarios.

## Preliminary 1: Life cycle of an OpenAI HTTP request
The life cycle of a single OpenAI HTTP request is as follows. Each step produces a single output.
1. "Request": A JSON payload representing a rollout is sent to the HTTP server endpoint.
   1. The schema here is typically in Responses input items or Chat Completions messages. For the rest of the sequence, we will assume Responses input items.
2. "Input prompt": The Responses input items are "chat templated" i.e. converted from objects into a single string.
3. "Prompt token IDs": The input prompt is "tokenized" i.e. converted from a string into a sequence of model-understandable token IDs.
4. "Generation token IDs": The prompt token IDs are sent to the model and the model generations a new sequence of token IDs.
5. "Generation": The generation token IDs are "de-tokenized" into a string.
6. "Response": The generation is "parsed" into a sequence of Responses output items.

We refer to the above life cycle steps 1-6 as LF1 - LF6.

## Preliminary 2: Life cycle of a rollout
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

# Preliminary 3: Train and generation time log probability mismatch



## Problem 1: Retokenization
Observe that going from LF5 of the previous model call to LF3 of the current model call (i.e. across model calls) implies that we lose information that was once held in token IDs.

For example, in the previous model call, the model may have produced token IDs 1 and 2 which were detokenized into "_Skinny" in LF5. Then in step LF3, we could re-tokenize "Skinny" into token ID 3.

Then, when we 
