# gymv_agent

Multi-turn text-action agent for the Gym-V resources server (`resources_servers/gym_v/`).

Runs a rollout against a Responses-API model server and Gym-V's `/step`
endpoint, extracting the model's action from the **last** `\boxed{...}` token
in the assistant's plain text.


## Context compaction

Requires the shared `nemo_gym.agents.context_management` API from
[Gym #3368](https://github.com/NVIDIA-NeMo/Gym/pull/3368), including the
per-observation omission-marker repair (RecencyHistoryPolicy v3). This adapter
does not include a private copy of that framework. Until the dependency lands,
this branch is a stacked draft, not a standalone main-compatible release.

Merge `configs/first_last_two.yaml` after the ordinary agent config, with the
same top-level agent name. It keeps the first and latest two reasoning blocks
and image observation groups, preserves instructions, feedback and all actions,
and limits the episode to 20 model calls. It reserves 8192 output tokens inside
a 51200-token total context guard. Set the request's `max_output_tokens`
explicitly to 8192 or less; the overlay does not change model sampling settings.
The initial observation is included before constructing the history client so
its images are protected. Image groups may contain multiple images.

The model template must also preserve the reasoning blocks selected by the
client. For Super templates that default to dropping historical thinking, set
`chat_template_kwargs.truncate_history_thinking: false` on the `vllm_model`
proxy and the matching serving-template defaults. Otherwise a prompt rebuilt
after compaction can lose all historical reasoning despite correct client
selection. Keep measurement and generation template settings identical. When
using NeMo-RL serving, provide explicit request sampling values matching its
configured profile; do not rely on model-generation defaults.

Token capture must be enabled and the dispatcher must supply the logical rollout
identity. The model server must support the capture-owned `/context/measure`
and `/v1/responses` routes. The client owns prompt compaction, measurement,
response acceptance and finalization. The adapter returns the full accepted
execution history and summed usage, not just the compacted model view.

- Omit `context_history` for the existing full-history path.
- Set `context_history.enabled: false` with valid guards for a capture-backed
  identity control using the same client.
- Model resampling and changing the per-call output budget are rejected.
- A measured guard rejection after a completed action ends at that real action
  with `termination_reason: context_length_exceeded`; an unseen observation is
  not published as a captured prompt. If the first call is rejected, the agent
  raises and discards the resource session's reward. It never fabricates a
  response or a scored zero. A typed pre-generation rejection envelope is not
  part of this adapter's pending dependency contract.
- A lost model acknowledgement or malformed response aborts the episode and
  discards its reward. It cannot return a previous successful response as if
  the uncertain call had not happened.

Tests cover enabled/disabled policies, 6- and 20-call histories, canonical
capture identity, image-only observation boundaries, guards, usage and real
HTTP disconnects. These are synthetic-response integration tests. Final live
model qualification must use the updated shared dependency; prior full-history
smoke runs do not qualify compaction.
