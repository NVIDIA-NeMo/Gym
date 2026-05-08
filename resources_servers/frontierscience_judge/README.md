# FrontierScience Judge

Single-pass LLM-judge resource server for free-form science olympiad answer
grading. Mirrors NeMo Skills' `frontierscience-olympiad` benchmark
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

The judge is invoked once per attempt (no two-order A/B comparison). Output
fields: `reward` (1.0 if `YES`, else 0.0), `verdict` (`YES`/`NO`/`null`),
`extracted_answer` (the model's post-thinking text), `judge_output` (the
judge's full text).

## Example usage

```bash
# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/frontierscience_judge/configs/frontierscience_judge.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts (5-example smoke test)
ng_collect_rollouts \
    +agent_name=frontierscience_judge_simple_agent \
    +input_jsonl_fpath=resources_servers/frontierscience_judge/data/example.jsonl \
    +output_jsonl_fpath=results/frontierscience_judge_rollouts.jsonl \
    +num_repeats=1
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
