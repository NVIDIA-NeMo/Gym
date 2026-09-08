# PinchBench

All 147 tasks of [PinchBench](https://github.com/pinchbench/skill) — calendar, email triage,
CSV/log analysis, coding, research, writing — measuring how well a model performs as the brain
of an [OpenClaw](https://github.com/openclaw/openclaw) agent.

This benchmark config is a thin narrowing of the `pinchbench` agent server, which owns the whole
integration (one sandbox per task, in-sandbox OpenClaw gateway, PinchBench's own grading harness).
Read [`responses_api_agents/pinchbench/README.md`](../../responses_api_agents/pinchbench/README.md)
first — architecture, the per-task image build, config knobs, and the parity validation against
vanilla standalone PinchBench all live there.

## Verification

Grading is PinchBench's own, run inside the sandbox: deterministic checks (`automated`), an LLM
judge (`llm_judge`), or a weighted mix of both (`hybrid`). Reward is **continuous** in `[0,1]`,
not pass/fail — threshold it if you need a binary signal. Each rollout also returns
`grading_type`, `grading_breakdown`, `grading_notes` and `raw_rollout`.

`num_repeats` is 3: temperature 1.0 plus live web search make a single pass noisy, and 3 is what
the agent-side parity validation used. Trust aggregates, not individual runs.

## Data preparation

```
gym eval prepare --benchmark pinchbench
```

The PinchBench skill is not vendored. `prepare.py` clones it at `v2.0.0` — the same ref
`responses_api_agents/pinchbench/Dockerfile.benchmark` bakes into the per-task image — and writes
`data/pinchbench_benchmark.jsonl`, one row per manifest task:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "<the task's ## Prompt section>"}]},
 "verifier_metadata": {"task_id": "task_sanity"}}
```

`task_id` is the authoritative selector: at run time `run_task.sh` passes it to
`benchmark.py --suite`, which loads the full task (prompt + assets + grading) from the skill inside
the sandbox. `input` carries the prompt for readability only. **Subset by dropping rows** — no
config change needed.

Set `PINCHBENCH_SKILL_DIR` to an existing `v2.0.0` checkout to skip the clone. Preparation fails
loudly if the manifest is not 147 tasks or if the regenerated rows do not byte-match the committed
`responses_api_agents/pinchbench/data/example.jsonl`, so upstream drift can't slip through.

## Quickstart

Build the per-task image first (`setup_scripts/build_image.sh --apptainer`), then:

```
gym eval run --benchmark pinchbench --model-type vllm_model \
    +sandbox_image=<pinchbench.sif | docker://pinchbench-openclaw:latest> \
    +model_base_url=<endpoint/v1> +model_api_key=<key> +model_name=<model> \
    +judge_model=<judge> +judge_base_url=<endpoint/v1> +judge_api_key=<key> \
    +brave_api_key=<key>
```

Those eight keys are declared in `config.yaml` with `null` defaults so `gym list` and
`gym eval prepare` resolve without them; each also reads a `PINCHBENCH_*` env var. A `null` still
reaching the agent fails fast — the fields are required in `PinchBenchAgentConfig`.

For slow or highly concurrent endpoints, raise `openclaw_provider_timeout_seconds` **and**
`task_timeout_s` together; see the agent README's "OpenClaw LLM idle timeout" section.
