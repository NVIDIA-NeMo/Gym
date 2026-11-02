# ChemEval

For preparation and evaluation through `gym eval`, use the [ChemEval benchmark integration](../../benchmarks/chemeval/README.md).

Gym integration of [ChemEval](https://github.com/USTC-StarTeam/ChemEval): 2,210 zero-shot text questions across 53 tasks, 13 capability dimensions, and levels L1-L4. The preparer excludes three-shot and multimodal questions; it does not expose a three-shot option.

## Automatic benchmark preparation

Use `gym eval prepare --benchmark chemeval` from the Gym checkout.
The benchmark folder now downloads and converts its pinned upstream sources.
No external evaluator installation, converted local input, or manual data copying is required.
See the [benchmark README](../../benchmarks/chemeval/README.md) for the current workflow.

The original dataset retains its source license (CC-BY-NC-4.0 on the dataset card;
the upstream repository declares CC-BY-NC-SA-4.0). Dataset files are downloaded at
preparation time and remain local and gitignored. Judge prompts are bundled locally.
The integration includes a standalone Apache-2.0 scoring implementation. `grader_path` optionally selects another
trusted Python scorer; the default is the bundled `grading.py`. A replacement
exports `grade(sample)`, which scores deterministic rows in place. English V2
judging is handled separately by the resource server.

## Prompts and scoring

Generation uses the user template: `{problem}{answer_format}`, with no added system message. Preparation removes restrictive no-explanation clauses from the problem. Original question text and gold answers remain in verifier metadata. Judged tasks use the bundled English V2 system message and a JSON user payload containing the question, complete final answer, reference answer (`ref-1`), and an empty list of verifier facts. Sampling parameters remain configurable separately from prompt contents.

The adapter strips think/thinking blocks and excludes separate reasoning output before grading. Structured answers use this extraction order: the last balanced object with an `answer` key, the last `Answer:` line, then the last nonempty line. Judged tasks use the complete final response.

MCQ extraction prefers the leading answer label, so explanatory text such as `B (a tropane)` does not override `B`. True/false extraction handles `not correct` and `not true` as negative verdicts. Entity extraction accepts JSON/Python lists as comma-separated entities while preserving task-specific aliases and normalization.

| Grading family | Reward |
|---|---|
| MCQ, true/false, classification | Selected letter or normalized label match |
| Classification subset | Gold label contained in the response, ignoring case |
| Entity extraction, relation extraction | Set F1 |
| Entity recognition | Per-question BIO token accuracy |
| Reagent selection | Set F1 over canonical SMILES |
| SIDER | Fraction of 20 labels correct |
| Molecule SMILES/SELFIES | Morgan fingerprint Tanimoto similarity |
| Molecular formula | Cosine similarity of element-count vectors |
| IUPAC name | Case-insensitive exact match |
| Numeric range | Interval intersection over union |
| Regression | `max(0, 1 - abs(predicted - gold) / gold_span)` |
| LLM judged: nine tasks, 450 zero-shot questions | `(legacy_score_1_to_5 - 1) / 4`; fill-in-the-blank uses `outcome.score_0_to_1` |

### English V2 judging

The judge text in [`english_judge.py`](english_judge.py) comes from the team's
`chemeval_english/improved_eval/chemeval_prompts_v2_en.py`, commit
`1f37bdf1206a52e28777628500f5e9742226eeda`. Eight task rubrics retain the original
English V2 messages; the molecular-description rubric is corrected as described
below. Only the judge portion is used;
candidate prompts and deterministic grading remain unchanged. This is a changed
judging protocol, so judged scores are not directly comparable with results from
the original Chinese rubrics.

| Prepared rubric | V2 task type |
|---|---|
| `fill_in_the_blank` | `fill_blank` |
| `short_answer` | `short_answer` |
| `calculation` | `calculation` |
| `abstract_generation` | `paper_abstract` |
| `outline_generation` | `research_outline` |
| `physicochemical` | `molecular_description` |
| `single_step_synthesis` | `single_step_synthesis` |
| `multi_step_synthesis` | `multi_step_synthesis` |
| `reaction_intermediate` | `reaction_intermediate` |

The 50 L3 `physicochemical` questions ask for descriptions of molecules from SMILES.
Their rubric evaluates reference-supported structural features, chemical classes,
properties, and roles, rather than systematic naming. It does not require naming
details or numerical properties absent from the question and reference. This
corrects the previous `molecular_name` mapping; previously judged scores for these
50 questions need rejudging to use the correction. Prepared inputs and the 1–5
score scale remain compatible.

The judge separates outcome correctness from visible process quality and requests
quoted evidence. The entire parsed JSON verdict is saved in
`judge_v2`; `judgement` and `judge_response` retain the raw response. Scoring
validates both numeric fields, not the full diagnostic schema: legacy scores
must be integers 1–5, and outcome scores must be finite numbers in [0, 1].
Surrounding prose or Markdown fences are tolerated. Invalid score fields produce
zero reward and `judge_parse_ok=false`, without retrying for a different verdict.

The system message requests JSON; no structured-output API mode or extra schema
is appended, matching the prior V2 requests. The model, reasoning effort, token
budget, and temperature remain configurable independently. Two released judged
calculation questions have empty reference answers; their payloads explicitly
use `(no reference answer supplied)`.

Regenerate older prepared inputs with `gym eval prepare --benchmark chemeval`.
Judged rows require `judge_protocol: english_v2`, `judge_question`, `judge_rubric`,
and the matching `judge_scale`; old Chinese prefix/suffix metadata is rejected.

Rewards preserve fractional credit. Raw diagnostic values are retained under `grading_metrics`, including numeric predictions/errors, F1, molecular similarities, and BIO counts. These per-question rewards do not reproduce upstream corpus RMSE or token-micro averages as the primary reward. Those can be recovered from the diagnostics.

Empty responses score zero without a judge call. Unparseable judge verdicts score zero with `judge_parse_ok=false`. Judge service/configuration errors use Gym's judge-failure handling. Malformed answers that raise numeric/literal errors score zero with `scoring_error`; nonfinite diagnostics become JSON null.

`compute_metrics()` averages repetitions within each question, questions within each named ChemEval task, and then tasks equally. `overall_score` is the mean of task scores, not the pooled mean over questions. It also reports scores by task, level, dimension, and grading family, plus `level_macro_score` and `num_tasks`. Generic Gym average reward remains question-weighted; use `overall_score` for the ChemEval headline metric. Scores use the 0-1 scale.

## Run

The benchmark config uses the policy model as judge. To use a separate
Responses-compatible judge, append `--config resources_servers/chemeval/configs/judge_model.yaml`
and supply `judge_base_url`, `judge_api_key`, and `judge_model_name` in your private
model configuration. A Chat Completions judge can instead use Gym's `vllm_model`
adapter. Apply judge/model overlays after the benchmark or resource config.

Use the model YAML shown in the [benchmark README](../../benchmarks/chemeval/README.md).
The first command runs servers in the foreground; run the second in another
terminal with the same Gym environment active.

```bash
gym env start --config benchmarks/chemeval/config.yaml \
    --model-type vllm_model --config /absolute/path/to/model.yaml

gym eval run --no-serve --agent chemeval_simple_agent \
    --input resources_servers/chemeval/data/example.jsonl \
    --output results/chemeval/example_rollouts.jsonl \
    --num-repeats 1 --concurrency 4 --max-output-tokens 16384 --temperature 0
```

The standalone resource config leaves `judge_model_server` unset; only use it without a judge for deterministic inputs. `grading_max_concurrency` defaults to 4; deterministic scoring runs in worker threads. `judge_max_concurrency` defaults to 32. The benchmark config sets judge temperature 0 and 16,384 output tokens through `judge_responses_create_params`; the standalone server uses the same 16,384-token default.

## Validation

```bash
gym env test --resources-server chemeval
gym dataset collate --config resources_servers/chemeval/configs/chemeval.yaml \
    --output-dir /tmp/chemeval-collated --mode example_validation
```

Tests use synthetic examples and cover the deterministic grading families, both
judge scales, preparation, error handling, and task-macro aggregation. They use
the bundled scorer; chemistry-dependent cases require the dependencies declared
in `requirements.txt`. The complete prepared input is
`benchmarks/chemeval/data/test.jsonl`. `verified: false` remains until baselining
and maintainer review; smoke tests alone do not certify benchmark fidelity.
