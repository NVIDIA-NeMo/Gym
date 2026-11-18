# Question-bank judge

Compose [question_bank.yaml](configs/question_bank.yaml), [dsv41_flash.yaml](configs/dsv41_flash.yaml), and the policy model configuration. Provide the judge endpoint, credential, model alias and pinned revision through the existing `STEM_JUDGE_BASE_URL`, `STEM_JUDGE_API_KEY`, `STEM_JUDGE_MODEL` and `STEM_JUDGE_MODEL_REVISION` settings.

The two agent names also work with previously prepared question-bank rows.

| Agent | How it grades the last boxed answer |
|---|---|
| `question_bank_math_simple_agent` | Math-Verify checks equivalence first. A nonmatch or additional grading instructions send the same answer to the model judge. |
| `question_bank_semantic_simple_agent` | The model compares the extracted answer with the question and reference using the shared science prompt. |

Both routes require a final `\boxed{...}`. An incorrect answer receives reward 0 and remains eligible for training. An inconclusive verification returns `verdict=unresolved`, `correctness=null`, `evaluation_completed=false`, and `mask_sample=true`. Its numeric zero is a placeholder. Repaired references, ambiguous questions, and verifier failures can be investigated using the recorded inputs, status, evidence and hashes. Known rubric, construction and native-test tasks are excluded before either route runs. The question-bank judge accepts ordinary question/reference rows.

The default [science prompt](prompt_templates/shared_answer_v2.txt) adapts Jiacheng Xu’s [STEM grading guidelines](https://github.com/NVIDIA-NeMo/Gym/blob/fbe250878cc317e893392a641070bb448fbd7fd9/resources_servers/stem_judge/prompt_templates/stem_v11.txt). The v1 template can be selected with `shared_prompt_template_fpath` for historical replay. Judge timeout and concurrency are configurable. A null `judge_timeout_seconds` defers the deadline to the serving layer.

## Prepare data

Run the adapter from the Gym repository root with `python -m resources_servers.question_bank_judge.prepare --help`. It emits Gym rows, the development and holdout splits, a selection ledger, and checksums. The solver receives the problem and the shared final-answer instruction. The judge receives the reference separately. Use `--prompt-overrides` for domain or source instructions during generation, and the `judge_domain_instructions` and `judge_source_instructions` configuration maps during verification. Preserve the original solver inputs when replaying historical answers.

When migrating from the experimental STEM branch, point the config paths and preparation command to `resources_servers/question_bank_judge`. Jiacheng’s `stem_judge_simple_agent` and rubric configuration live in his independent STEM resource.
