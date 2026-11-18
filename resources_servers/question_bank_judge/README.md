# Question-bank grading

Prepared rows select a verifier by domain. The two agent names are the same ones used by earlier question-bank exports.

| Question-bank domain | Agent | Verifier |
|---|---|---|
| Mathematics | `question_bank_math_simple_agent` | Gym’s built-in `math_with_judge`. Math-Verify checks equivalence first. On a nonmatch with a boxed answer, Gym uses its standard binary math prompt. A positive fallback judgment requires agreement with the answer order reversed. |
| Biology, chemistry / materials, engineering / computer science, medicine / health, physics, humanities / social sciences, other / unlabeled | `question_bank_semantic_simple_agent` | `question_bank_judge` compares the final boxed answer with the question and reference using its semantic comparison prompt. It returns `correct`, `incorrect`, or `unresolved`. |

On the semantic route, a completed incorrect answer receives an unmasked zero. Inconclusive verification returns `verdict=unresolved`, `correctness=null`, `evaluation_completed=false`, and `mask_sample=true`. On the math route, inspect `library_reward` and `judge_evaluations`. A fallback reply with no parseable verdict carries `judge_parsing_error` and scores zero under Gym’s math verifier.

## Configuration

Both verifiers use the same `judge_model` endpoint. The training policy uses a separate `policy_model`. Compose the policy model config with [question_bank.yaml](configs/question_bank.yaml) and [dsv41_flash.yaml](configs/dsv41_flash.yaml). If the training launcher already supplies a `judge_model` config, use that endpoint config in place of `dsv41_flash.yaml`.

For the bundled endpoint config, set `STEM_JUDGE_BASE_URL` to the deployment’s OpenAI-compatible API base URL, `STEM_JUDGE_API_KEY` to its credential and `STEM_JUDGE_MODEL` to its served model name. Set `STEM_JUDGE_MODEL_REVISION` to the deployed checkpoint revision for either configuration. The semantic verifier masks requests that reach model comparison without a configured revision.

The default [semantic comparison prompt](prompt_templates/shared_answer_v2.txt) adapts [Jiacheng Xu’s STEM grading guidelines](https://github.com/NVIDIA-NeMo/Gym/blob/fbe250878cc317e893392a641070bb448fbd7fd9/resources_servers/stem_judge/prompt_templates/stem_v11.txt). Its subject-specific examples illustrate which differences preserve the answer the question asks for. The same equivalence and verdict rules apply across eligible domains. Jiacheng’s `stem_judge_simple_agent` and rubric configuration live in his independent resource. To replay earlier semantic judgments, select the v1 template with `shared_prompt_template_fpath` and preserve the solver input the model received. The semantic judge timeout and concurrency are configurable. A null `judge_timeout_seconds` defers the deadline to the serving layer.

## Prepare data

Run `python -m resources_servers.question_bank_judge.prepare --help` from the Gym repository root. The adapter emits Gym rows, development and holdout splits, a selection ledger, and checksums. Humanities / social sciences and other / unlabeled are recognized dataset labels. Their answer-comparison rows use the semantic judge. Rubric, construction, native-test and unrecognized-domain exclusions appear in the ledger. If an older export sends a row with a missing or ineligible domain to the semantic server, it returns a masked `unsupported_domain` result before calling the model.

The solver receives the problem and the shared final-answer instruction. The verifier receives the reference separately. Use `--prompt-overrides` for domain or source instructions during generation. The semantic verifier also accepts `judge_domain_instructions` and `judge_source_instructions`. Earlier experimental math scores used a different fallback and should be identified by the verifier revision when comparing runs.

Gym’s default training collation accepts the question-bank provenance fields on math rows. Strict task-data validation flags `id`, `uuid`, and `metadata` as undeclared by the built-in math schema. For strict math collation, use `task_id` for the identifier and keep the remaining provenance in the preparation ledger.
