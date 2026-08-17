# physics_judge

Physics QA resource server with NeMo Skills' physics judge as the LLM-judge
fallback for math-verify symbolic checking. The judge is asked
"is this output correct?" and emits `[Correct]` / `[Incorrect]` verdict tokens.

Subclasses `math_with_judge` (`LibraryJudgeMathResourcesServer`) and overrides
four pieces of behaviour:

1. **Judge prompt** — loaded from `prompts/judge.yaml` at server init via
   Gym's prompt system (`load_prompt_config`). The YAML must define a
   `user` key (Skills-style placeholders: `{problem}`, `{generation}`,
   `{expected_answer}`); a `system` key is optional. The bundled prompt is
   a character-for-character copy of NeMo Skills'
   `nemo_skills/prompt/config/judge/physics.yaml`. To swap in a different
   judge prompt, override `judge_prompt_path` in the server config.
2. **Verdict tokens `[Correct]` / `[Incorrect]`**, matched
   case-insensitively, mirroring Skills'
   `PhysicsMetrics.is_correct_judgement` regex semantics.
3. **Single judge call** — Skills' physics judge has a fixed
   `Question / Output sentence / Correct answer` role assignment, so the
   bidirectional A/B swap done by `math_with_judge` is skipped.
4. **Per-domain breakdown** — `compute_subset_metrics(field="domain")` is
   run alongside the regular `compute_pass_majority_metrics`, so each
   physics domain shows up as `<domain>/pass@1[avg-of-k]/...` keys in the
   metrics JSON. Mirrors Skills' `subset_for_metrics=domain`.

`math_with_judge` itself is untouched, so this server has zero impact on
existing math_with_judge consumers (`aime24`, `aime25`, `gsm8k`,
`hendrycks_math`, etc.).

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --resources-server physics_judge \
    --resources-server physics_judge/judge_openai
```

The bundled `judge_openai.yaml` defaults the judge to `openai/gpt-oss-20b`
on `https://integrate.api.nvidia.com/v1` and reads the API key from
`NVIDIA_API_KEY`. To use a different judge, override any of the
`judge_model.responses_api_models.openai_model.openai_*` fields on the
CLI, or replace `judge_openai.yaml` with your own config that defines a
`judge_model:` server.

> Reasoning-model note: start the policy vLLM server with
> `--reasoning-parser deepseek_r1` (or the model-specific parser).
> That strips `<think>…</think>` at the model edge, so `\boxed{…}`
> extraction and the judge both see clean post-think text.

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent physics_judge_simple_agent \
    --input resources_servers/physics_judge/data/example.jsonl \
    --output results/physics_judge_rollouts.jsonl \
    --num-repeats 1
```

## Metrics

Inherits `compute_metrics()` and `get_key_metrics()` from `math_with_judge`,
extended with a per-domain breakdown via `compute_subset_metrics`:

- `pass@k/symbolic_accuracy` and `pass@1[avg-of-k]/symbolic_accuracy` —
  symbolic correctness (math-verify pass rate)
- `pass@k/judge_accuracy` and `pass@1[avg-of-k]/judge_accuracy` — judge
  pass rate, computed only on rollouts that fell through to the judge
- `majority@k/...` — majority-vote accuracy (uses `extracted_answer`)
- `<domain>/pass@1[avg-of-k]/...` — per-domain pass@k for benchmarks
  that include a `domain` field per row (e.g. PHYSICS).
