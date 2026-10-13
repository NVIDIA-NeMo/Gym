# FrontierScience Judge

Single-pass LLM-judge resource server for FrontierScience grading. It supports
two modes:

- `judge_mode: olympiad` for FrontierScience-Olympiad free-form answer
  equivalence.
- `judge_mode: research` for FrontierScience-Research 10-point rubric scoring,
  with `reward = 1.0` when the parsed score is at least
  `rubric_pass_score_threshold` (`7.0` by default).

The default mode mirrors NeMo Skills' `frontierscience-olympiad` benchmark
verification pipeline:

- The judge sees the problem, attempted answer, and reference answer in a
  single prompt and returns `Judgement: YES` or `Judgement: NO` on its
  final line.
- The verdict is parsed by anchoring on the last `Judgement:` occurrence
  (matching Skills' `is_correct_judgement`).
- The default judge prompt is verbatim Skills'
  `nemo_skills/prompt/config/judge/frontierscience-olympiad.yaml` (sourced
  from the [FrontierScience paper](https://cdn.openai.com/pdf/2fcd284c-b468-4c21-8ee0-7a783933efcc/frontierscience-paper.pdf)
  page 13). The path is configurable via `judge_prompt_path` so other
  free-form-grading benchmarks can reuse this server with their own prompt.
- Research mode uses `prompts/research_judge.yaml`, which asks the judge for
  `Score: X/10` and `Judgement: YES/NO`.

The judge is invoked once per attempt (no two-order A/B comparison). Output
fields: `reward` (1.0 if `YES`, else 0.0), `verdict` (`YES`/`NO`/`null`),
`extracted_answer` (the model's post-thinking text), `judge_output` (the
judge's full text), and in research mode `rubric_score` plus
`rubric_score_normalized`.

## Example usage

```bash
# Running servers
gym env start \
    --model-type vllm_model \
    --resources-server frontierscience_judge

# Collecting rollouts (5-example smoke test)
gym eval run --no-serve \
    --agent frontierscience_judge_simple_agent \
    --input resources_servers/frontierscience_judge/data/example.jsonl \
    --output results/frontierscience_judge_rollouts.jsonl \
    --num-repeats 1
```

The default `judge_model` config points at the public NVIDIA inference
API (`https://integrate.api.nvidia.com/v1`) and `openai/gpt-oss-20b`,
reading the key from `NVIDIA_API_KEY`. To swap in a different judge
endpoint — for example, the original Skills configuration of
`o3-mini-2025-01-31` via `api.openai.com` — override the top-level
`judge_base_url` / `judge_api_key` / `judge_model_name` vars in
`configs/frontierscience_judge.yaml`.

For Nemotron-3-Nano and other reasoning models, start vLLM with
`--reasoning-parser deepseek_r1` so `<think>...</think>` is stripped at
the model edge — the server's verdict parsing assumes the answer text
follows the closing `</think>` tag.

## Verification flow

```
model output ──reasoning-parser──> generation
                                        │
                                        ▼
       judge_prompt = template.format(question=…, expected_answer=…, generation=…)
                                        │
                                        ▼
                              call judge model (single pass)
                                        │
                                        ▼
                            parse last "Judgement: YES|NO"
                                        │
                                        ▼
                              reward = 1.0 if YES else 0.0
```
