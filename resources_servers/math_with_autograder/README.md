# math_with_autograder

Math QA resource server with a Skills-style **autograder LLM judge** as the
fallback for math-verify symbolic checking. The judge is asked
"is this answer Correct or Incorrect?" rather than "are these two answers
equivalent?" — the prompt provides the role of grader, the model answer
is the "Model Solution", and the expected answer is the "Golden Answer".

Subclasses `math_with_judge` and overrides three pieces of behaviour:

1. **Judge prompt** — loaded from `prompts/judge.yaml` at module import. The
   YAML must define a `user` key (Skills-style placeholders: `{problem}`,
   `{predicted_answer}`, `{expected_answer}`); a `system` key is optional.
   This server's bundled prompt is a copy of NeMo Skills'
   `nemo_skills/prompt/config/judge/imo_answerbench.yaml`.
2. **Raw `\boxed{...}` to the judge** — on a math-verify miss, the judge
   sees the literal LaTeX inside the model's last balanced `\boxed{...}`
   rather than math-verify's normalized form. math-verify is tuned for
   numeric / algebraic answers and silently mangles non-numeric answers
   (sets, function definitions, conditions) into a degenerate fragment.
3. **Single judge call** — the autograder template has a fixed Model /
   Golden role assignment, so the bidirectional swap done by
   `math_with_judge` is skipped.

To plug in a different autograder prompt (e.g. for a non-IMO benchmark),
subclass `MathWithAutograderResourcesServer` and override:

- `JUDGE_PROMPT_TEMPLATE` (point at a different `prompts/<name>.yaml`)
- `JUDGE_EQUAL_LABEL` / `JUDGE_NOT_EQUAL_LABEL` (if the prompt emits
  different verdict tokens)

`math_with_judge` itself is untouched, so this server has zero impact on
existing math_with_judge consumers.

## Run servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/math_with_autograder/configs/math_with_autograder.yaml,resources_servers/math_with_autograder/configs/judge_gptoss20b.yaml"
ng_run "+config_paths=[$config_paths]"
```

The bundled `judge_gptoss20b.yaml` wires the autograder judge to
`openai/gpt-oss-20b` via NVIDIA's public NIM API (reads `NVIDIA_API_KEY`
from the environment). Override with your own `judge_model` config to
swap to a different OpenAI-compatible endpoint.

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=math_with_autograder_simple_agent \
    +input_jsonl_fpath=resources_servers/math_with_autograder/data/example.jsonl \
    +output_jsonl_fpath=results/math_with_autograder_rollouts.jsonl \
    +num_repeats=1
```

## Metrics

Inherits `compute_metrics()` and `get_key_metrics()` from `math_with_judge`,
so this server emits the same headline metrics:

- `pass@k/symbolic_accuracy` and `pass@1[avg-of-k]/symbolic_accuracy` —
  symbolic correctness (math-verify pass rate)
- `pass@k/judge_accuracy` and `pass@1[avg-of-k]/judge_accuracy` — judge
  pass rate, computed only on rollouts that fell through to the judge
- `majority@k/...` — majority-vote accuracy (uses `extracted_answer`)
