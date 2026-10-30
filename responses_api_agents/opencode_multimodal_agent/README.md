# OpenCode multimodal agent

Run image-based tasks through OpenCode and collect their responses, tool results,
intermediate images and usage in Gym. The agent supports training with token
capture and evaluation with configurable turn limits. A Gym resources server
scores each task.

The implementation extends `opencode_agent` and uses OpenCode **1.17.11**.
Each episode stages its input images in a private workspace. The supplied
configurations run OpenCode inside a Bubblewrap sandbox. Session artifacts are parsed into Gym responses
and trajectory records before the workspace is cleaned up.

## Requirements

Use a Linux host with:

- The native OpenCode 1.17.11 executable on `PATH`.
- Bubblewrap with permission to create user namespaces.
- A Python environment containing Pillow, numpy, scipy and OpenCV for tool use.
- Preinstalled `@opencode-ai/plugin` dependencies at version 1.17.11.
- A Gym model server and a resources server implementing task verification.

Install this agent's Python requirements from its directory:

```bash
uv pip install -r requirements.txt
```

The dependency directories are mounted read-only inside each episode. Input
images are available as `input-0.<extension>`, `input-1.<extension>`, and so on
under `/testbed`. Training converts inputs to PNG; evaluation preserves their
format and also supports the `image_` filename prefix.
OpenCode's `read` tool displays images saved by code; printing a filename does
not display the image to the model.

For a disposable local runtime, follow the
[multimodal setup instructions](../../scripts/harness_conformance/README.md#run-diagnostic-probes).
That setup also checks that OpenCode and the image libraries work inside the sandbox.

## Configure the agent

Start with [the agent configuration](configs/opencode_multimodal_agent.yaml).
Supply the model identifier, model and resources server references, and paths to
these runtime assets:

```yaml
sandbox:
  bwrap_command: [/usr/bin/bwrap]
  python_env: /opt/visual-runtime/python
  plugin_dependencies: /opt/visual-runtime/plugins
```

`plugin_dependencies` must contain `node_modules/@opencode-ai/plugin` and an
initialized OpenCode configuration directory. The setup helper prepares this
layout. Use the native executable rather than npm's Node launcher in the sandbox.

Choose the agent and model entrypoints together:

| Purpose | Agent entrypoint | Model entrypoint |
|---|---|---|
| Training with token capture | `app.py` | `app_opencode.py` |
| Training with a custom system prompt | `combined.py` | `app_opencode.py` |
| Evaluation | `eval.py` | `visual_eval.py` |

The base configuration supplies an 8-turn budget, a 49152-token context and
output ceiling, and a 2048-token final-answer reserve. These are configurable
values, not requirements of the harness. Set `temperature` and `top_p` explicitly
in each task's `responses_create_params`; the agent requires both. Configure the
serving model's context capacity and sampling options to agree with the task and
agent settings.

The agent exposes `/v1/responses` for generation and `/run` for generation plus
verification. It manages image staging and sandbox lifetime for the whole episode;
separate agent-session setup and close operations are not supported.

## Prompts

Two configuration fields have different roles:

- `system_prompt` is a **user-message prefix**, despite its name. The base agent
  defaults to the guidance in `prompts/visual_prefix.txt`. Set it to `null` to omit
  that prefix.
- `build_system_prompt`, available in `combined.py` and `eval.py`, replaces the
  OpenCode **system prompt**. Evaluation requires `system_prompt: null` when this
  override is set. `combined.py` defaults to `prompts/combined.txt` and always
  omits the user-message prefix.

With no system-prompt override, OpenCode selects its own prompt. Task text and
image attachments are supplied separately from the system prompt.

## Turn and token limits

For training, `max_turns` and `final_answer_reserve` control a plugin that requests
a final answer and disables tools when the budget is reached. The paired
`app_opencode.py` model adapter handles OpenCode's version-specific final-turn
instruction before token lineage resolution and capture. See the
[model adapter documentation](../../responses_api_models/vllm_model/README.md#stock-opencode-11711-training).

Exact-token reservation requires backend support for `nemo_final_answer_reserve`.
With a backend that does not implement it, set `backend_final_answer_reserve: false`
and keep `final_answer_reserve` positive. The plugin still enforces the final-answer
turn and budgets output using the previous response's reported usage. Without
backend support, that estimate cannot account exactly for new tool-result tokens.
Setting `final_answer_reserve: null` disables the entire plugin and uses OpenCode's
native advisory step behavior; it does not enforce the training turn limit.

For evaluation, configure the same `turn_budget` on the agent and model gateway
to enforce a hard limit. For example:

```yaml
turn_budget:
  regular_turns: 55
  grace_turns: 10
  grace_tool_turns: 2
  grace_tool_calls: 2
```

Set the agent's `max_turns` to the sum of regular and grace turns: 65 in this
example. The final-answer notice starts on turn 55. Turns 56 and 57 can each
execute at most one additional tool call, subject to the total grace allowance.
The remaining grace turns disable tools. Requests beyond turn 65 are rejected
before another model generation. This mode requires one model-gateway worker,
a rollout ID and the chat-completions API. Turns count assistant calls, not
individual tools; a regular turn can contain several tool calls.

Without `turn_budget`, evaluation uses OpenCode's advisory step threshold and
validates the resulting trajectory against `max_turns`. The
`final_answer_turn_headroom` setting reserves room between that threshold and
the trajectory limit; its default is 2. This mode does not provide the gateway's
hard generation stop. Set both `turn_budget` and `max_turns` to `null` for an
uncapped evaluation episode; context and timeout limits still apply.

`context_window` and `max_output_tokens` bound model requests. Also configure the
backend's matching token limits. `timeout` limits the episode and
`command_exec_timeout` limits a shell command. Evaluation's `compaction_auto`
controls automatic OpenCode compaction.

## Images and reasoning history

The agent's `max_images` limits initial attachments. Requests exceeding it fail
before generation. `max_source_pixels` limits the size of an input image.
`image_policy` controls attachment resizing; the base policy resizes images
larger than 8000 pixels in either dimension.

The model adapter's separate `max_images` setting limits images accumulated in
model requests. Additional images are replaced with:

> Image limit reached. This additional image was omitted. You cannot receive additional images.

This guard operates on a request copy and does not modify the saved history.
Set the model limit to `null` to disable it. Evaluation also accepts `null` for
the agent's initial-image limit. Match these settings to the serving model's
image capacity.

Reasoning-history rendering is controlled by the model server's
`chat_template_kwargs.truncate_history_thinking`; set it to `false` to retain
thinking. Exact-token prefix restoration during training is handled by the
training backend, so message rendering alone does not guarantee identical
training and inference token sequences.

## Verification and observability

`/run` sends the final answer to the selected resources server and retains the
full response for inspection. `strip_reasoning_for_grading` removes embedded
thinking from the scoring copy without altering the saved response. Optional
`tool_rewards` adds tool-based reward shaping; set it to `null` to use only the
verifier reward.

Reward shaping is configured under `tool_rewards`:

- `correct_only` gates all auxiliary adjustments on
  `correct_answer_reward_threshold`; it defaults to `false`.
- `correct_bash_call_penalties` supplies a nonpositive schedule indexed by total
  bash attempts, including failed and duplicate attempts. Index zero must be
  zero, and the last entry applies to all larger counts. The default `[0.0]`
  disables this adjustment.
- `invalid_tool_call_names` restricts failure penalties to selected tools;
  `null` includes all tools and an empty list disables failure penalties.
  Error counts still include failures outside the selected tools.
- Correct-answer success bonuses, exploration rewards and their cap, failure
  penalties, and consecutive-duplicate penalties are separately configurable.
  The failure-tool filter does not restrict duplicate penalties.

Shaping uses native execution records from the main trajectory; replayed and
child-agent calls do not earn auxiliary rewards. A successful tool execution
is not evidence of a correct answer: the resources server supplies that score.

Training uses `token_id_capture` to retain data for optimization. Evaluation can
disable training token capture while retaining model-call observability.
`capture_message_ids` correlates OpenCode assistant messages with model calls.
`capture_failure_artifacts` preserves available session evidence on failures.
Recognized agent failures in the evaluation entrypoint produce zero-scored
results with failure metadata. Judge and unclassified infrastructure errors
remain errors rather than fabricated answers.

For datasets and benchmark verification, see the
[visual benchmark guide](../../benchmarks/opencode_multimodal_benchmarks/ONBOARDING.md).
For scripted harness checks, run the `opencode_multimodal` conformance suite using
the [conformance instructions](../../scripts/harness_conformance/README.md).
