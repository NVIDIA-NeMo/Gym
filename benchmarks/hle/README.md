# HLE Benchmark

Benchmark wrapper for [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle), a
2158-question (text-only subset) exam covering graduate-level STEM and humanities knowledge.

- **Tasks**: 2158 text-only questions (image questions filtered at prepare time)
- **Reward**: binary; LLM judge checks whether the model's response matches the ground-truth answer
- **Metrics**: `pass@1/judge_accuracy` — fraction of questions judged correct

The judge uses the official HLE evaluation prompt adapted from
[`centerforaisafety/hle`](https://github.com/centerforaisafety/hle), which extracts the model's
final answer and checks it against the expected answer with a yes/no verdict. The policy model
serves as the judge — no separate judge server is needed.

## Dataset access

`cais/hle` is a gated HuggingFace dataset. Request access at
[https://huggingface.co/datasets/cais/hle](https://huggingface.co/datasets/cais/hle), then
authenticate:

```bash
huggingface-cli login
```

## Prepare benchmark data

```bash
gym eval prepare --benchmark hle
```

Downloads `cais/hle`, filters to text-only questions, and writes
`benchmarks/hle/data/hle_benchmark.jsonl`.

### Vision (multimodal) subset

HLE's image questions are exposed as a variant config in this directory,
`config_vision.yaml`, i.e. the benchmark `hle/config_vision`:

```bash
gym eval prepare --benchmark hle/config_vision
```

This downloads the full `cais/hle` split (2500 questions: the same 2158 text
questions plus 342 image questions) and writes
`benchmarks/hle/data/hle_benchmark_vision.jsonl`. Unlike the text-only file,
these rows are **fully materialized** — the prompt template is baked into
`responses_create_params.input`, and image questions carry an `input_image`
block (base64 data URI). Because the input is pre-populated, this dataset uses
`prompt_config: null` (the two are mutually exclusive; a prompt template can
only produce string content, so an image block has to be built at prepare
time).

The `include_vision` flag lives on `prepare.py`; `prepare_vision.py` is a thin
wrapper that calls `prepare(include_vision=True)`. Running the text-only
prepare directly with the flag also works:

```bash
python benchmarks/hle/prepare.py --include-vision
```

Evaluating the vision variant requires a vision-capable policy model:

```bash
gym env start --model-type vllm_model --benchmark hle/config_vision
```

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark hle
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in
`env.yaml` (or passed as CLI overrides).

For the vision variant use `--benchmark hle/config_vision` (agent
`hle_vision_equivalence_llm_judge_simple_agent`) with a vision-capable policy
model.

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent hle_equivalence_llm_judge_simple_agent \
    --input benchmarks/hle/data/hle_benchmark.jsonl \
    --output results/hle_rollouts.jsonl \
    --prompt-config benchmarks/hle/prompts/default.yaml \
    --num-repeats 1 \
    --temperature 0.0
```

Use `temperature: 0.0` to match the nemo-skills evaluation setup and ensure reproducible scores.

For the vision variant, drop `--prompt-config`: those rows already carry
`responses_create_params.input`, and passing a prompt config would overwrite it
(the run fails with "Some rows have responses_create_params.input but
prompt_config is also specified").

```bash
gym eval run --no-serve \
    --agent hle_vision_equivalence_llm_judge_simple_agent \
    --input benchmarks/hle/data/hle_benchmark_vision.jsonl \
    --output results/hle_vision_rollouts.jsonl \
    --num-repeats 1 \
    --temperature 0.0
```

## Metrics

`pass@1/judge_accuracy` is the headline metric.

## Sandboxed agents

Use `--benchmark hle/opencode` or `hle/pi` for Python tools without search, or
`hle/opencode_search` or `hle/pi_search` to enable Tavily search as well.
These text-only presets preserve standard preparation and grading, with the
Explanation/Answer/Confidence instructions in a single user message and one repeat.
Keep collection repeats at one.

`hle/opencode` and `hle/opencode_search` run `opencode_agent` through this benchmark's `single_agent_turn_legacy`
Environment Server. Each session creates its own sandbox from `OPENCODE_SANDBOX_IMAGE`, the offline science image
built from [`offline_science_image`](../../responses_api_agents/opencode_sandboxed_agent/offline_science_image), which
reaches only the model server (and the Tavily server for search). The search preset grants the agent the Tavily
server's MCP tools for each episode. The shared settings are in
[`offline_science.yaml`](../../responses_api_agents/opencode_agent/configs/profiles/offline_science.yaml).
`hle/opencode_legacy_agent` and `hle/opencode_search_legacy_agent` keep the deprecated `opencode_sandboxed_agent`
on the `legacy_agent` relay.

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

`hle/pi` and `hle/pi_search` run `pi_agent` the same way, from `PI_SANDBOX_IMAGE`, the image built from
[Pi's `offline_science_image`](../../responses_api_agents/pi_sandboxed_agent/offline_science_image), with the shared
settings in [`offline_science.yaml`](../../responses_api_agents/pi_agent/configs/profiles/offline_science.yaml).
`hle/pi_legacy_agent` and `hle/pi_search_legacy_agent` keep the deprecated `pi_sandboxed_agent` on the
`legacy_agent` relay.

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

The search presets supply the public HLE [exclusion list](search_exclusions.json).
Deployments requiring an additional policy must supply their combined policy file
through the Tavily resource's `exclude_domains_file_path`; keep private policies
outside the public repository. This list reduces answer exposure but cannot
guarantee that every answer-bearing page is excluded.
