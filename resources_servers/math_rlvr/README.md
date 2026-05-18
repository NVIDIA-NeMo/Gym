# math_rlvr Resources Server

Port of NeMo-RLVR's `nemo_rl/environments/math_environment.py` to NeMo-Gym.

### Overview

A single math verifier with **per-row dispatch** over four verification types,
selected via the `verifier_type` field on each request:

| `verifier_type` | What it does | Backed by |
|---|---|---|
| `math` (default) | Wrap `ground_truth` in `\boxed{...}` and call `math-verify`'s `math_metric` (LaTeX/SymPy-aware). | `math-verify==0.8.0` |
| `math500` | Strict `\boxed{...}` extraction on both sides + Hendrycks MATH `is_equiv` normalizer (whitespace/`\frac`/`\sqrt`/units). | vendored `math_utils.py` |
| `english_multichoice` | Regex `Answer:\s*([A-Z])` + `normalize_response`/`normalize_extracted_answer`. | vendored `answer_parsing.py` |
| `multilingual_multichoice` | Walk a bank of multilingual `Answer:` regex prefixes (Korean, Bengali, Chinese, Arabic, Spanish, French, …) and normalize the captured letter to A-D. | vendored `answer_parsing.py` |

Pure scoring — no tools, no LLM judge, no sandbox. A `<think>...</think>`
block is stripped before grading; an unclosed `<think>` (no `</think>`)
yields reward 0.0 with `verification_failed=False`.

### Why this is a separate server from the existing math servers

Gym already has several math servers; this one exists because RLVR's
`math_environment.py` covers a different surface area:

| Server | Distinguishing feature |
|---|---|
| `math_rlvr` (this) | Per-row `verifier_type` dispatch covering MATH500-style boxed grading **and** English / multilingual MCQA, all in one server. |
| `math_with_judge` | `math-verify` library with an LLM-judge fallback for ambiguous cases. |
| `math_with_code` | Math grading paired with a Python execution tool. |
| `math_advanced_calculations` | Multi-step tool use (calculator, SymPy) for advanced calculations. |

Pick `math_rlvr` when porting an RLVR dataset that already encodes its
`verifier_type` (e.g. mixed MATH500 + multilingual MMLU); pick the others
when their distinguishing feature is what you actually need.

### Input schema

Each row carries (alongside `responses_create_params` and `response`):

- `ground_truth` (required, str) — the expected answer text. Shape varies by
  `verifier_type`:
  - `math`: a LaTeX expression to wrap in `\boxed{...}` (e.g. `"42"`,
    `"\frac{1}{2}"`).
  - `math500`: a string containing a `\boxed{...}` block (e.g.
    `"\boxed{120}"`).
  - `english_multichoice` / `multilingual_multichoice`: a single uppercase
    letter `A`-`D`.
- `verifier_type` (optional, str; default `"math"`) — one of `math`,
  `math500`, `english_multichoice`, `multilingual_multichoice`. Unknown
  values fall back to `math`.

### Verification response fields

In addition to the base `reward`:

- `verifier_type` — echoed back exactly as requested (even for unknown values
  that fell back to `math`).
- `extracted_answer` — the extracted answer where applicable (`math500`,
  multichoice). `None` for the `math` verifier (math-verify's library output
  is not surfaced).
- `verification_failed` — `True` when the verifier itself raised an
  unexpected exception (defensive). `False` for ordinary 0.0 rewards and
  for unclosed-`<think>` cases.

### Example dataset row

```json
{
  "verifier_type": "math500",
  "ground_truth": "\\boxed{120}",
  "responses_create_params": {
    "input": [{"role": "user", "content": "Compute 5!. Put your final answer inside \\boxed{}."}],
    "tools": [],
    "parallel_tool_calls": false
  }
}
```

### Usage

```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_rlvr/configs/math_rlvr.yaml"
ng_run "+config_paths=[$config_paths]" \
  "+simple_agent.responses_api_agents.simple_agent.resources_server.name=math_rlvr"

ng_collect_rollouts \
  +agent_name=simple_agent \
  +input_jsonl_fpath=resources_servers/math_rlvr/data/example.jsonl \
  +output_jsonl_fpath=results/math_rlvr_rollouts.jsonl \
  +num_repeats=5
```

### Testing

```bash
ng_test +entrypoint=resources_servers/math_rlvr/
```

The `math-verify` library is required (pinned to `0.8.0` to match
`math_with_judge`); `ng_test` builds an isolated venv from this server's
`requirements.txt` so SymPy and friends are picked up correctly.

### Differences from the RLVR original

- **`followup` / teacher-feedback retry is NOT ported.** RLVR's
  `MathEnvironment.step()` optionally calls an OpenAI-compatible "teacher"
  model to give hints when the answer is wrong, and continues the rollout
  with `done=False` so the student can retry. That is multi-turn behavior;
  in Gym it belongs in an agent under `responses_api_agents/`, not on the
  resources server. Port the verifier first; layer a follow-up agent on top
  later if you want it.
- **`dapo_math_verifier` is dropped** — RLVR imports `compute_score as
  dapo_math_verify` but never actually calls it from `step()`. Removing
  the import.
- **The Ray worker pool, `chunk_list_to_workers`, `EnvironmentReturn`,
  `additional_rewards`, and `global_post_process_and_metrics`** are dropped
  per Gym conventions.

### Vendored files

- `math_utils.py` — subset of NeMo-RLVR's `nemo_rl/evals/math_utils.py`
  used by the `math500` verifier (`last_boxed_only_string`, `remove_boxed`,
  `is_equiv`, and the `strip_string`/`fix_*` helpers). The original module's
  `process_docs`/`process_results` (which depend on `datasets`) are
  intentionally omitted.
- `answer_parsing.py` — full copy of NeMo-RLVR's
  `nemo_rl/evals/answer_parsing.py`.

Both keep the original Apache-2.0 header.

### Licensing

Code: Apache 2.0. Vendored verifier helpers are reproduced from NeMo-RLVR
(Apache 2.0).
