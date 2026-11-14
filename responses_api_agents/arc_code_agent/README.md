# ARC code agent

A multi-turn, image-aware, code-executing agent harness for ARC-style grid puzzles, built for vision-language
reasoning models served by vLLM (OpenAI-compatible, reasoning parser on). It is a port of a harness first written for
raw-text Qwen3.8 runs; this version talks to the Gym model server through the Responses API.

## What the agent does

1. Builds the whole prompt from the row (`examples`, `test_input`; the row's own `input` is ignored): a system prompt, then
   one user message with a computed "Observations" block and, for every grid, its digits followed by an image.
2. Calls the model. The model reasons, then writes Python in `<code>...</code>` (generation stops at `</code>`), or calls a
   `python` function tool when `native_tools` is on (see below).
3. Runs the code in a sandbox that keeps state between runs and returns the output as a user turn
   `<tool_response>\n...\n</tool_response>`.
4. Repeats until the model writes the final grid in `<answer>...</answer>`. A turn cut by the per-turn token cap gets a
   harness nudge. When the token budget is low, one more call forces the final answer with thinking off.
5. Calls the resources server `/verify` (exact match of the final grid, reward 1.0 or 0.0).

The first run whose `check(transform)` passes every example pair ends the attempt (`auto_answer`): the checked
function's prediction for the test input becomes the answer, recorded as a synthetic assistant message.

## Native tool calling (`native_tools: true`)

Some models are trained for native function calling and do not write `<code>` blocks. With `native_tools: true` the
agent offers one function tool, `python` (argument `{"code": "<python source>"}`, `strict: false`), in every request,
sets no stop string, and the system prompt's code-running sentence says to call the `python` tool; everything else in
the prompt (helpers, budget, run count, answer format) is unchanged. The model server must parse tool calls (vLLM
`--enable-auto-tool-choice` with the model's tool parser) and use a reasoning parser.

- Each `function_call` item of a turn is run in the sandbox, in order, and answered with a `function_call_output`
  item with the same `call_id` whose output is the sandbox text (stdout, then `[error] ...`), without the
  `<tool_response>` wrapper. Several calls in one turn share the sandbox state and each counts as a run.
- Arguments that are not JSON, lack a non-empty string `code`, or a call to another tool name, return an
  `[error] ...` output (and count as a run). Calls beyond `max_runs` are answered with a "not run" output and, with
  `runs_cap_final`, the next request is the forced final answer. A call in a turn cut by the token limit is answered
  with a "not run" output, followed by the usual cut nudge.
- A turn with neither a function call nor an answer follows the text protocol: `nocode_nudge` turns are nudged,
  otherwise the attempt ends. Typed answers, `answer_resend`, `auto_answer` and the token budget work as before.
- The forced final answer keeps the tool definition with `tool_choice: "none"` and thinking off.
- Reasoning and function calls are replayed as `reasoning` and `function_call` items (`keep_thinking`), which the
  model server turns into one assistant message with its reasoning and tool calls.
- `response.output` returned by the agent is the whole history after the prompt: `reasoning`, `function_call`,
  `function_call_output`, messages, nudges and the synthetic auto-answer message.

## Row schema

`{"task_id", "examples": [{"input", "output"}, ...], "test_input", "expected_output", "responses_create_params": {...}}`,
grids as `list[list[int]]`. One test input per row.

## Returned record

`response.output` holds the full conversation after the prompt: the model's reasoning and messages, the tool turns, harness
nudges and the synthetic auto-answer message. `responses_create_params.input` is replaced by the exact prompt that was
sent (system message plus the user message with images as `data:image/png;base64,...` URLs; set `record_images: false`
to store `<image omitted>` instead). Per-rollout fields on the verify response: `turns`, `code_runs`, `cuts`,
`resends`, `generated_tokens`, `used_tokens`, `prompt_tokens`, `wall_seconds`, `finished_with_answer`, `answer_source`
(`typed`, `auto`, `forced` or `none`), `forced_final`. Aggregate metrics use Gym's standard reward profile, which
averages these numeric fields.

## Configuration

`configs/arc_code_agent.yaml` lists every field with its default; the defaults reproduce the original setting
(xhigh effort, 73,728-token cap, 4,096 answer tokens, 24 runs, 336 px images at scale 12 with the pad32 style,
temperature 0.7, 30 s sandbox timeout, 4,000 output characters). Required: `resources_server` and `model_server`.
The model server needs `policy_base_url`, `policy_api_key` and `policy_model_name` in `env.yaml` as usual.

The agent sends per-call settings through request metadata that `vllm_model` understands: `chat_template_kwargs`
(`{"enable_thinking": true, "reasoning_effort": "<effort>"}` for thinking turns, `{"enable_thinking": false}` for
the forced answer) and `extra_body` (`top_k`, `seed`, `stop`, `include_stop_str_in_output`, plus the `extra_body`
config field). If the model's chat template uses other kwarg names, set `thinking_template_kwargs` and
`final_template_kwargs`.

```bash
gym env start \
    --config resources_servers/arc_code/configs/arc_code.yaml \
    --config responses_api_agents/arc_code_agent/configs/arc_code_agent.yaml \
    --model-type vllm_model
gym env test +entrypoint=responses_api_agents/arc_code_agent
```

## Sandbox

A pool of `num_sandbox_workers` long-lived Python servers; each code run forks a child that applies limits, executes the
puzzle header, silently replays the episode's earlier code blocks (so state persists without long-lived children) and
runs the new block with stdout captured, echoing a final bare expression. Preloaded helpers: `check`, `predict`,
`objects`, `diff`, `crop`, `bbox`, `bg`, `colors`, `variants`, `norm`, `find`, `holes`, `regions`, `inside`, `panels`,
`runs`, `ray`, `canon`, `fits`, plus common names (`ndimage`, `Counter`, `itertools`, ...).

Isolation it gives: wall-clock timeout (`timeout_s`, child killed at the deadline), CPU, address-space (`mem_mb`),
file-size and open-file limits, output truncation (`out_chars`), a fresh scratch directory per run, a scrubbed server
environment (no inherited secrets), no `open`/`input`/`exit`, and an import allowlist that excludes `os`,
`subprocess` and `socket`. A timed-out cell is discarded from the replay (`drop_timed_out`).

What it does not give: the restricted builtins are not a security boundary (a determined program can escape them), there
is no network or filesystem namespace by default, and it needs `os.fork` (Linux/macOS). For untrusted models run the
agent in a container, and/or set `sandbox_command_prefix` (for example `["unshare", "-rn"]` for no network, or a
bubblewrap command).

## Ported from the harness (all active in the default setting)

Image rendering identical to the original (same palette, nearest-neighbour integer scale, pad32 canvas; checked
byte-for-byte against the original renderer on real puzzles); prompt text built by the same sentence swaps (budget
and run count, image note, helper list, sandbox libraries, output cap); computed Observations; digits plus image per
grid; stop at `</code>`; code extraction including unterminated blocks and native `<tool_call>` blocks
(`parse_tool_calls`); tool turns; past turns re-sent with reasoning (`keep_thinking`); per-turn cap with the cut nudge
and `max_cuts`; total budget with a forced final answer and thinking off, `clean_final`, `runs_cap_final`;
`auto_answer`; `answer_resend` of unreadable answers; `drop_timed_out`; `tool_fixes`; helper sets v2 and v3_new;
`sandbox_hygiene`; `out_chars`, `timeout_s`; per-row request seeds (`seed_mode`, `seed_offset`); optional
`nocode_nudge` (off by default, as in the original setting, where a turn with neither code nor answer ends the attempt).

## Changes forced by running on Gym

- The sandbox variable and prompt wording say `examples` (the original said `train`), and "example pair" for "training
  pair"; the vendored helper code was renamed accordingly. Prompt wording is otherwise unchanged.
- Transport: Responses API through the Gym model server instead of a raw chat endpoint. Reasoning arrives as separate
  output items (reasoning parser on) and is replayed as such. The tool turn is a user message with the
  `<tool_response>` wrapper (the same text a chat template renders for a tool message).
- Token accounting: the original counted harness text with a tokenizer; the agent has none. It uses the model server's
  reported prompt size (exact) plus a `chars_per_token` estimate for text added since the last call.
- Cut detection: the original read the finish reason; Gym responses may not carry one, so a turn counts as cut when
  the response is marked incomplete or its output tokens reach the per-turn limit.
- The request seed uses the rollout index Gym passes (`_ng_rollout_index`) instead of a sample number.
- Sandbox processes are managed with asyncio; the original used threads. The server script gained a per-run scratch
  directory.

## Deliberately dropped

Everything tied to the raw-text completions path: building `<|im_start|>`/`<think>` text by hand, the `raw` and `chat`
observation styles, token-exact prompt reuse, and the tokenizer. Test-time fine-tuning and anything about its
competition notebook runtime. The command-line driver, sharding, resume files, concurrency admission gate and result
files. Experiment options that the default setting leaves off: alternative prompt layouts and image styles (JFP
canvas, pair images, test-only images, diff images, image-only prompts), history trimming and notes, plan turns, review
turns, augmented-copy / `expect` / coverage / test-probe checks, check and prediction images, fail notes, stage and
budget nudges, verification and require-check rounds, give-up answers, warm starts, attempt symmetries, answer gates and
voting pools, optional extra prompt paragraphs, `helpers_v3_fix` and `split_grid`. Multi-test merging (`per_task`) is
dropped because the row schema has a single test input; a task with several test inputs is several rows.
