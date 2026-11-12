# APEX Shortlist

Math problems from MathArena's APEX Shortlist, sourced from
`MathArena/apex-shortlist` on HuggingFace. Mirrors the NeMo Skills
`apex-shortlist` benchmark (`nemo_skills/dataset/apex-shortlist/`).

## Verification

Reuses the `math_with_judge` resource server in **symbolic-only** mode
(`should_use_judge: false`) to mirror NeMo Skills' `eval_type=math`
default for this benchmark. The HuggingFace `math-verify` library does
symbolic equivalence of the model-extracted `\boxed{...}` answer against
`expected_answer`.

## Prompt

User-only prompt, character-for-character match with NeMo Skills'
`generic/math.yaml`:

```
Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}.

<question>
```

## Data preparation

```bash
gym eval prepare --benchmark apex_shortlist
```

Writes `data/apex_shortlist_benchmark.jsonl` with one row per problem:
`{"task_id": "...", "question": "...", "expected_answer": "..."}`, where `task_id` is the dataset's `problem_idx`.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark apex_shortlist
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent apex_shortlist_math_with_judge_simple_agent \
    --input benchmarks/apex_shortlist/data/apex_shortlist_benchmark.jsonl \
    --output results/apex_shortlist_rollouts.jsonl \
    --num-repeats 4
```

## Sandboxed agents

Use `--benchmark apex_shortlist/opencode` or `apex_shortlist/pi` for Python tools
without search. These presets preserve the standard boxed-answer prompt and
symbolic grading, with 16 dataset repeats per question.
Keep collection repeats at one.

`apex_shortlist/opencode` runs `opencode_agent` through this benchmark's `single_agent_turn_legacy` Environment
Server. Each session creates its own sandbox from `OPENCODE_SANDBOX_IMAGE`, the offline science image built from
[`offline_science_image`](../../responses_api_agents/opencode_sandboxed_agent/offline_science_image), which reaches
only the model server. The shared settings are in
[`offline_science.yaml`](../../responses_api_agents/opencode_agent/configs/profiles/offline_science.yaml).
`apex_shortlist/opencode_legacy_agent` keeps the deprecated `opencode_sandboxed_agent` on the `legacy_agent` relay.

`opencode_agent` cannot express everything the deprecated OpenCode preset set, so scores of the two are not one
measurement:

- Output budget: 32,768 tokens per model call within a 229,376-token context. OpenCode asks for the output cap on
  every call, so the deprecated preset's `remaining_context` policy has no equivalent, and asking for the whole
  262,144-token window makes vLLM refuse the call.
- Compaction, the step limit and session titles follow OpenCode's defaults: `opencode_agent` accepts only
  `permission` and `tools` in `opencode_config`, so the deprecated preset's no-compaction setting and 400-step cap
  do not apply.
- Failures: a run that stops at its output, context or time limit is graded on what it produced. A harness crash
  (nonzero exit, no result) is retried and then reported as a failed rollout. The deprecated preset scored both
  as 0 (`execution_failure_reward_zero`).
- Commands: `opencode_agent`'s default rules (destructive commands, Git network and history mining) apply, and
  subagents stay denied.
- Requests: `opencode_agent` accepts the input, `instructions`, `model`, `temperature` and `top_p`, and refuses
  other request fields, such as `reasoning` or `max_output_tokens`, with 422; the deprecated agent ignored them.
- Artifacts: there is no host export directory (`OPENCODE_ARTIFACTS_DIR`); the rollout rows carry the agent's
  observations.
- The image ships no plugin-dependency bundle for `opencode_agent`'s private config directory, so each offline
  session start waits about 70 s for OpenCode's install attempt to fail. Build one as the
  [`opencode_agent` README](../../responses_api_agents/opencode_agent/README.md#pinned-offline-runtime-bundle)
  describes and set `sandbox_plugin_dependencies_path` and `sandbox_plugin_dependencies_sha256` to remove the wait.

`apex_shortlist/pi` runs `pi_agent` the same way, from `PI_SANDBOX_IMAGE`, the image built from
[Pi's `offline_science_image`](../../responses_api_agents/pi_sandboxed_agent/offline_science_image), with the shared
settings in [`offline_science.yaml`](../../responses_api_agents/pi_agent/configs/profiles/offline_science.yaml).
`apex_shortlist/pi_legacy_agent` keeps the deprecated `pi_sandboxed_agent` on the `legacy_agent` relay.

`pi_agent` carries over the deprecated Pi preset's runtime, output policy, compaction, Bash cap and network policy.
What still differs:

- Failures: a run that stops at its output, context or time limit is graded on what it produced. A harness crash
  (nonzero exit, no result) is retried and then reported as a failed rollout. The deprecated preset scored both
  as 0 (`execution_failure_reward_zero`).
- Prompt: `pi_agent`'s runtime guard adds a short note on sandbox execution to the system prompt.
- Requests: `pi_agent` refuses request fields other than the input, `instructions` and `model`, such as
  `temperature` or `reasoning`, with 422; the deprecated agent ignored them.
- Artifacts: there is no host artifact directory (`PI_ARTIFACTS_DIR`); the rollout rows carry the agent's
  observations.

See the shared [OpenCode and Pi setup guide](../../responses_api_agents/opencode_sandboxed_agent/README.md#offline-scientific-evaluation-with-opencode-or-pi)
for images, credentials and network policies. Its token limits, step cap and artifact directories describe the
`*_legacy_agent` presets.
