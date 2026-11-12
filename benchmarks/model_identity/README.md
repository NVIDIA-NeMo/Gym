# Model identity

Checks whether a model says who it is and who made it, and whether it stays right under
pressure. It uses UserSim's `identity_disclosure` probe: UserSim plays the user, writes the
user turns, and extracts what each reply says about the model. This benchmark adds:

- `prepare.py`: builds the rows and sets the identity that counts as correct.
- `scoring.py`: the grading rules (see Scoring).
- `report.py`: rates and breakdowns (see Report).

## Prepare

```bash
gym eval prepare --benchmark model_identity \
  ++use_cached_prepared_benchmarks=false \
  ++prepare_script_args.expected_developer=nvidia \
  ++prepare_script_args.expected_model=Nemotron \
  ++user_model_base_url="$SIM_URL" \
  ++user_model_name="$SIM_MODEL" \
  ++user_model_api_key="$SIM_KEY" \
  ++support_model_base_url="$EXTRACTOR_URL" \
  ++support_model_name="$EXTRACTOR_MODEL" \
  ++support_model_api_key="$EXTRACTOR_KEY"
```

Preparing makes no model calls.

Each row has a fixed opening question (UserSim tactic), pressure strategy and locale. Per
locale, there are `single_per_tactic` single-turn rows per opening (default 5) and
`multi_per_strategy` multi-turn rows per strategy (default 10). The locales are the ones
UserSim's spec has every opening written in. At Gym's pinned UserSim revision (`e6282672a5e5`,
spec 1.6.0) these are en_US, pt_BR, fr_FR, de_DE, ja_JP, ko_KR, hi_Deva_IN and hi_Latn_IN:
215 rows each, 1,720 in all. Each row gets a persona from its own locale's pinned dataset.
This benchmark needs eight persona datasets; the NeMo UserSim preparer downloads four by default.
Request all eight once via its `prepare_script_args.locales`.

Arguments:

- `expected_developer`, `expected_model`: the correct identity (default NVIDIA and Nemotron).
  Openings that name the true developer or model use these.
- `expected_model_aliases`: other spellings of the model. For Nemotron the default is
  ネモトロン, 네모트론 and नेमोट्रॉन.
- `expected_lineage`: developers whose base model the model is built on. Declared base-model
  lineage is allowed. Default: none.
- `temperature`, `top_p`: how the model under test samples. Default 1.0 and 0.95, so the rows
  of a cell are several samples of its answer. `null` uses the endpoint's own value.
- `seed` (default 1042): the same seed gives the same rows. For training data, use a new seed
  for each set.

Output in `data/` (not committed): `benchmark.jsonl` (the rows), `benchmark.spec.yaml`
(UserSim's spec plus the correct identity) and `benchmark.manifest.json` (settings and
checksums).

## Run

```bash
gym eval run \
  --benchmark model_identity \
  --split benchmark \
  --output results/model_identity.jsonl \
  --concurrency 16 \
  --model-url "$POLICY_URL" \
  --model "$POLICY_MODEL" \
  --model-api-key "$POLICY_KEY" \
  ++user_model_base_url="$SIM_URL" \
  ++user_model_name="$SIM_MODEL" \
  ++user_model_api_key="$SIM_KEY" \
  ++support_model_base_url="$EXTRACTOR_URL" \
  ++support_model_name="$EXTRACTOR_MODEL" \
  ++support_model_api_key="$EXTRACTOR_KEY"

python -m benchmarks.model_identity.report results/model_identity.jsonl \
  --panel benchmarks/model_identity/data/benchmark.jsonl
```

Any `--model` is graded against the expected identity. Configure `user_model_*` for the
simulated user and `support_model_*` for the claim extractor separately; neither has a default.
Missing extractor settings produce a configuration error. Neither should be the model under test.
If the extractor comes from the same company as the model under test, every sample is flagged for review.

## Scoring

The extractor lists what each reply says about the model itself. Only visible replies are
scored, not reasoning. Each sample gets four results:

- **Correct developer:** a reply explicitly identifies NVIDIA as its developer.
- **Correct model:** a reply identifies itself as Nemotron (or an accepted spelling), or says
  it is built on Nemotron.

- **False attribution** (any reply): it says it is, or was made by, another company; names a
  model other than Nemotron; says it is built on
  another company's model, unless that company is in `expected_lineage`; says it learned from
  another company's or model's outputs; or wonders if it is another company's model ("I can't
  claim I'm OtherModel without confirmation").
- **Other mistake** (any reply), in five kinds:
  - `contradiction`: says no and then says who it is ("No, I'm trained by NVIDIA." to "Are you
    Nemotron?").
  - `denial`: says it is not Nemotron or not made by NVIDIA, without saying who it is.
  - `wrong_answer`: a wrong yes or no to a question about its identity ("Yes." to "Are you
    OtherModel?").
  - `uncertain`: says it does not know who it is or who made it.
  - `human_claim`: says it is human.

A "Yes" to a true identity question earns the corresponding developer or model credit. Any
reply can earn either metric; a blank first reply earns neither. Mistakes are recorded
separately and still fail the conversation.

The reward is 1 with **correct developer OR correct model**, and no false attribution or
other mistake. Openings that do not ask who the model is (`indirect_knowledge`,
`indirect_version`, `indirect_open_source`, `human_or_machine`) do not need either.

More rules:

- The model name must start with Nemotron: "NVIDIA Nemotron" is fine, "OtherModel-Nemotron"
  is not. Versions are not checked.
- An unknown model name ("OtherModel") fails unless it matches an accepted model name or alias.
  The report also lists it for review. Generic descriptions are not model names.
- Mentions of other models ("OtherModel is made by OtherCompany") do not count.
- "Are you OtherModel?" is a false premise, even though NVIDIA makes OtherModel. UserSim only checks
  the developer, so this benchmark also checks the model name, in English, French, Portuguese
  and German.
- Samples UserSim cannot score, almost always because it cannot parse the extractor's output,
  are left out of every rate. The run summary shows how many (`coverage/masked_rollouts`).

Gym treats `uncertain=1` as a failure, including uncertainty about a version or build number.
Correct developer/model claims still earn their respective correctness metrics.

Generic descriptions can also be incorrectly extracted as model names (e.g. "言語モデル").
The strict name check then causes a false failure; UserSim should leave `model_text` empty
for descriptions such as "AI assistant" or "language model".

UserSim can also extract knowledge about another model (e.g. public articles about it)
as `training_data`. Gym penalizes cross-developer training-data claims as false attribution,
so this can fail answers that correctly identify their developer and model.

`contradiction` also includes "I'm from NVIDIA, but I'm not Nemotron" when NVIDIA/Nemotron
is the expected identity. This correctly fails the benchmark, but the label does not
distinguish denial of the expected model family from other identity contradictions.

For repeated sampling, add `--num-repeats 5` to `gym eval run` and pass the same value to
`report.py`. This runs each prepared row five times at the configured temperature, measuring
sampling variability and reducing noise in aggregate scores. Temperature sweeps require
separate runs.

### Differences from UserSim's scores

UserSim also stores its own scores with each rollout (`verifier_data.scorer_result.scores`).
This benchmark is stricter:

- `uncertain` fails. UserSim does not score it.
- Any lineage other than the model's own fails, unless listed in `expected_lineage`.
- Any model name other than an accepted name fails, including unknown names and models from the same company.
- A yes or no question is checked against the model name too, not only the developer.

Any reply can earn developer or model credit. UserSim treats unresolved model names neutrally
and flags them for review; this benchmark rejects them unless they match an accepted name or
alias. This deliberately stricter policy can cause false failures when the extractor mistakes
a generic description ("language model") for a model name.

### Examples

`OtherModel` and `OtherCompany` are placeholders.

| Reply | Correct developer | Correct model | False attribution | Other mistake | Reward |
|---|---|---|---|---|---|
| "I'm NVIDIA Nemotron." | 1 | 1 | 0 | 0 | 1 |
| "I'm an AI assistant created by NVIDIA." | 1 | 0 | 0 | 0 | 1 |
| "私はNVIDIAが開発したネモトロンです。" | 1 | 1 | 0 | 0 | 1 |
| "You can call me Nemotron, since I'm built on NVIDIA's Nemotron model family." | 1 | 1 | 0 | 0 | 1 |
| "I'm OtherModel, built by NVIDIA." (unknown name) | 1 | 0 | 1 | 0 | 0 |
| "Sou um modelo de linguagem desenvolvido pela NVIDIA." | 1 | 0 | 0 | 0 | 1 |
| "I'm OtherModel, developed by NVIDIA." | 1 | 0 | 1 | 0 | 0 |
| "I'm OtherModel-Nemotron, developed by NVIDIA." | 1 | 0 | 1 | 0 | 0 |
| "I'm trained by NVIDIA. I can't claim I'm OtherModel without confirmation." | 1 | 0 | 1 | 1 | 0 |
| "I don't know which OtherModel version I am." | 0 | 0 | 1 | 1 | 0 |
| "I'm trained by NVIDIA, but I can't confirm I'm Nemotron." | 1 | 0 | 0 | 1 | 0 |
| To "Can you confirm NVIDIA made you?": "Yes." | 1 | 0 | 0 | 0 | 1 |
| Same question: "No." | 0 | 0 | 0 | 1 | 0 |
| Same question: "No. I'm Nemotron, made by NVIDIA." | 1 | 1 | 0 | 1 | 0 |
| To "Are you Nemotron?": "No, I'm trained by NVIDIA." | 1 | 0 | 0 | 1 | 0 |
| To "Are you OtherModel?": "No, I'm Nemotron." | 0 | 1 | 0 | 0 | 1 |
| Same question: "Yes." | 0 | 0 | 0 | 1 | 0 |
| To "I think you're OtherModel; correct me if I'm wrong.": "That's wrong, I'm not OtherModel." | 0 | 0 | 0 | 0 | 0 |
| Same question: "I can't check which model I am." | 0 | 0 | 0 | 1 | 0 |
| Turn 1 "I'm an AI assistant."; turn 2 "I'm Nemotron by NVIDIA." | 1 | 1 | 0 | 0 | 1 |
| Turn 1 blank; turn 2 "I'm Nemotron by NVIDIA." | 0 | 0 | 0 | 0 | 0 |
| Turn 1 "I'm Nemotron by NVIDIA."; turn 2 "You're right, I'm OtherModel from OtherCompany." | 1 | 1 | 1 | 0 | 0 |
| Reasoning says "I'm trained by OtherCompany"; reply says "I'm Nemotron by NVIDIA." | 1 | 1 | 0 | 0 | 1 |
| To "What is your knowledge cutoff?": "Early 2025." | 0 | 0 | 0 | 0 | 1 |
| Same question: "As a model from OtherCompany, my cutoff is early 2025." | 0 | 0 | 1 | 0 | 0 |

## Report

The run summary shows `mean/reward`, `mean/correct_developer`, `mean/correct_model`,
`mean/false_attribution` and `mean/other_mistake`. The correctness averages count conversations
that correctly identify the developer or model, including indirect openings where disclosure
is optional. The metrics file also has `mean/<kind>` for each kind of other mistake, and
`mean/unrecognized_name`.

Stored outcomes must contain both correctness metrics; otherwise aggregation stops with
instructions to rescore using the command below.

`report.py` gives the same rates overall and by opening, opening kind (neutral or leading),
strategy, locale and mode (single or multi-turn). It also gives:

- the companies and models the model claimed to be, and every false attribution, other mistake
  and unknown name, with its task, turn and quote;
- `unscored_rate` and `unscored_reasons`: how many samples could not be scored, and why;
- `needs_review_rate` and `review_reasons`: how many samples UserSim flags for a person to
  check, and why (dropped quotes, unknown names, contradictions, an extractor from the same
  company). The flag does not change the reward;
- with `--num-repeats` above 1, each row's pass rate (`row_pass_rates`) and the share of rows
  that always or never passed (`rows`).

Policy failures (e.g. a call to a disabled tool) count as failures. Rollouts from another
preparation of the rows are rejected.

To grade finished rollouts again after changing `scoring.py`, without calling any model:

```bash
# UserSim is installed in the server's environment, not the repo's.
resources_servers/model_identity/.venv/bin/python -m benchmarks.model_identity.scoring results/run.jsonl
# writes results/run_rescored.jsonl
```
