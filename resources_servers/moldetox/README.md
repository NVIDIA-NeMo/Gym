# MolDeTox benchmark environment

[Dataset](https://huggingface.co/datasets/MolDeTox/MolDeTox) (CC-BY-4.0) and
[paper](https://arxiv.org/abs/2605.12181).

Toxicity-aware molecular editing, built on *toxicity cliffs* — pairs of
structurally similar molecules with opposite toxicity labels for the same
endpoint. Given a toxic molecule and an endpoint, the model must produce the
minimally-edited non-toxic partner.

The dataset revision is pinned to `059e1926fa62a07c3f01afadadf72cce154cfb7f`.
Upstream revised the files on 2026-08-27 and split sizes changed, so the name
alone is not a stable reference.

## Scope

All eight QA configs are implemented. The ninth upstream config,
`toxicitycliff`, is the source pair table rather than a task; it is used only to
join the toxic input molecule onto each row.

| Config | Test rows | `scoring_mode` | Model must produce |
| --- | --- | --- | --- |
| `task1_single` | 1,026 | `fragments_task1` | The one fragment driving toxicity |
| `task1_multi` | 950 | `fragments_task1` | Two or more toxic fragments |
| `task2_single` | 943 | `fragments_task2` | The one replacement fragment |
| `task2_multi` | 1,033 | `fragments_task2` | Two or more replacement fragments |
| `task3_smiles_gen_single` | 943 | `smiles` | Whole detoxified molecule, as SMILES |
| `task3_smiles_gen_multi` | 1,033 | `smiles` | Same, multi-fragment edit |
| `task3_safe_gen_single` | 943 | `safe` | Whole detoxified molecule, as SAFE |
| `task3_safe_gen_multi` | 1,033 | `safe` | Same, multi-fragment edit |

`single` and `multi` are a property of the item — how many fragments must change
— not a multi-turn protocol. Every config is one-shot.

Every verify response carries a `status`. A zero reward that is a fault in the
task row rather than a judgement on the model also carries a `failure_reason`,
so the two can be told apart downstream:

| Status | Meaning | `failure_reason` |
| --- | --- | :---: |
| `scored` | Compared against the gold answer | no |
| `empty_output` | No assistant text in the response | no |
| `no_answer` | No `{"answer": ...}` object found | no |
| `answer_too_long` | Answer exceeds the character cap | no |
| `unsupported_mode` | `scoring_mode` is not a known mode | yes |
| `bad_gold` | Gold answer missing or unusable for the mode | yes |
| `bad_metadata` | A `verifier_metadata` field has the wrong type | yes |

The last three return a controlled zero instead of raising, so one malformed row
cannot fail a whole eval with an HTTP 500. `endpoint` and `dataset_name` are
provenance labels: a wrong type there drops the label and still scores the row.

Only the `test` split is used. The dataset also ships a `train` split of about
202,000 rows, which nothing here reads.

## Prompting

Both turns come from upstream. The dataset ships each complete user message in
`question`, which `prepare_moldetox.py` passes through verbatim, and the paper
publishes a system prompt per task family as Appendix Tables H, I, J and K,
which `prompts.py` carries verbatim and the preparer prepends.

The system prompts restate the JSON contract the `question` already names, but
add role framing and a HARD CONSTRAINTS block forbidding explanations, markdown
and extra keys. An earlier version of this server sent no system message on the
grounds that the `question` was already complete. It is not the same input, and
the difference is measurable, so `--no-system-prompt` exists to reproduce the
omission rather than to support it.

**This is zero-shot, which is upstream's own baseline reporting mode**, so the
prompt shape matches the unlabelled rows of the paper's result tables. The
paper's 4-shot columns are a separate labelled variant that retrieves four
structurally similar examples from the train split; nothing here does that.

Prompt shape is only half of a comparable run. Section 3.2 also fixes the
sampler: **temperature 0.7, three runs per model, reported as mean ± standard
deviation**. A run that leaves temperature unset is not comparable to a published
figure no matter how faithful the prompt is, and this server sets no sampler
defaults of its own — that is the caller's to match. `top_p` and the token limit
are not stated upstream, so an exact reproduction cannot be built from the paper
alone.

## Dataset format

Each prepared row carries:

- `responses_create_params.input` — two turns: the task family's system prompt
  from `prompts.py`, then the upstream `question` verbatim as the user message
- `verifier_metadata.answer` — gold answer; a SMILES molecule, a SAFE string, or
  a dot-separated fragment set depending on the config
- `verifier_metadata.scoring_mode` — `smiles`, `safe`, `fragments_task1` or
  `fragments_task2`; selects how `answer` is compared and which auxiliary
  metrics are produced
- `verifier_metadata.toxic_smiles` — the toxic input molecule, joined from the
  `toxicitycliff` config on `source_index`. Carried but not read; it is PRS's
  reference molecule, and PRS is not implemented
- `verifier_metadata.common_safe_fragments` — Task 2 configs only. Upstream ships
  it as a field but does not put it in the prompt; its intended role is
  unresolved, so it is carried and unused
- `verifier_metadata.endpoint`, `dataset_name` — echoed on the verify response
  for stratified reporting
- `verifier_metadata.config`, `task`, `variant`, `id` — provenance
- `agent_ref` — `{"type": "responses_api_agents", "name": "moldetox_simple_agent"}`

`task_data.py` holds the same schema as a pydantic model with per-field notes on
what reads each one.

## Scoring

Reward is exact match in every mode. What "exact" means differs:

- **`smiles`** — canonical SMILES equality via RDKit.
- **`safe`** — the prediction is decoded to SMILES with the datamol SAFE decoder,
  then compared canonically. A failed reconstruction is counted invalid, matching
  upstream, rather than being masked.
- **`fragments_task1`, `fragments_task2`** — string multiset over the
  dot-separated SAFE fragments. Order does not matter, multiplicity does.
  Individual SAFE fragments carry unmatched attachment digits (`Cc1cccc6c1`,
  `C46`) and **do not parse as molecules**, so canonicalisation is impossible
  here and comparison is verbatim.

The two fragment modes share that reward and differ only in auxiliary metrics,
because Appendix D defines different ones per task. Appendix D.1 gives Task 1
exact match and F1 only; Appendix D.2 adds a fragment-level edit distance for
Task 2. Emitting a Task 1 distance would publish a number upstream never
reports, so the mode, not a second field, decides what is produced.

Additional metrics ride along on the verify response:

| Mode | Auxiliary metrics |
| --- | --- |
| `smiles`, `safe` | `validity`, `tanimoto_rdk`, `tanimoto_maccs`, `tanimoto_morgan`, `levenshtein` |
| `fragments_task1` | `fragment_f1`, `n_fragments_predicted` |
| `fragments_task2` | `fragment_f1`, `n_fragments_predicted`, `levenshtein` |

Two definitions follow the paper rather than the obvious reading. `fragment_f1`
is computed over fragment **sets**, as Appendix D.1 writes it, so a repeated
fragment neither helps nor hurts. The Task 2 `levenshtein` is **not** an edit
distance over the whole SAFE string; Appendix D.2 takes each predicted
fragment's minimum distance to any gold fragment and averages those minima.

Metrics a mode does not produce are left `None` rather than 0.0, so an absent
metric is never averaged in as a zero. `endpoint` and `dataset_name` are echoed
for stratification.

Only `mean/reward` and the token counts are promoted to `key_metrics`. The
similarity metrics stay in `agent_metrics` for diagnosis and are deliberately
kept off the headline line, for the reason in the next section.

### Two departures from the upstream metric set

- **BLEU-1 is not computed.** Upstream reports it as bare clipped unigram
  precision, which is degenerate on SMILES — the invalid string `C1CC` scores a
  perfect 1.0 against a 35-character gold, since every character occurs in it.
  Adding a brevity penalty fixes that but then the figure no longer matches how
  upstream computes it, so the metric is dropped rather than reported in a form
  that is either misleading or incomparable.
- **`levenshtein` is `None` when the prediction is not a usable molecule**, so
  those rows are excluded from the mean rather than charged the full gold length.
  This matters most in `safe` mode, where a prediction that does not decode has
  no string to compare at all; charging it the full gold length inflates the mean
  well above what upstream reports.

### Two things to know before reading a score

- **Similarity metrics cannot substitute for accuracy.** Echoing the toxic input
  unchanged scores 0.0 accuracy at mean Morgan Tanimoto **0.711** across all 943
  rows of `task3_smiles_gen_single`, because cliff pairs are similar by
  construction. A do-nothing baseline looks strong on every similarity metric.
- **PRS is not implemented.** Appendix D.3 names the six properties (MW, logP,
  HBA, HBD, PSA, RotB) and states the reference molecule is the toxic input, but
  leaves the per-property desirability functions and their weights unstated as
  "QED-inspired", and there is no upstream implementation to check against.
  Those curves determine the value entirely. `toxic_smiles` is carried in
  `verifier_metadata` so PRS can be added without re-preparing data.

## Harness validation

No model is involved in either check.

- **Gold-as-prediction.** Feeding every gold answer back through `verify()`
  scores `reward=1.0` on **7,904 of 7,904 rows** across all eight configs, every
  row status `scored`.
- **Negative controls.** Echoing the toxic input scores zero on all 943 rows of
  `task3_smiles_gen_single`; an undecodable SAFE string scores `validity=0`; an
  invalid SMILES is scored rather than silently dropped; and a renamed or
  reordered fragment set behaves as the multiset rules require.

A verifier that cannot fail would be worthless, so the negative controls matter
as much as the positive one.

No model evaluation results are recorded here. They are Surveyor-side evidence
and live with the benchmark's implementation notes.

## Setup

This server needs its own virtual environment; `rdkit` and `safe-mol` are not in
Gym's base install, and `gym` is not on `PATH` without it.

Both versions are pinned. `safe-mol` 0.2.1 supports RDKit 2025.09 but excludes
2026.03, which loses stereochemistry, so `rdkit` is held at 2025.9.6 rather than
the newer release.

Data preparation uses `huggingface-hub` directly rather than `datasets`. The
eight configs are plain `.jsonl` files at a pinned revision, and both paths were
compared over all eight and produce identical rows, so the direct loader is
simpler and deterministic. **It does not reduce the installed footprint**:
`nemo-gym` declares `datasets` in its own base dependencies, so `datasets`,
PyArrow and Pandas are installed either way. Dropping it removes a duplicate
direct declaration and the dataset-builder code path, nothing more.

```bash
cd resources_servers/moldetox
uv venv --python 3.13.14
uv pip install -r requirements.txt
cd ../..
source resources_servers/moldetox/.venv/bin/activate
```

**Every command below runs from the repository root with that environment
active.**

## Quickstart

The example file needs servers started separately; `gym eval run` accepts
`--input` only together with `--no-serve`, and `--split example` is rejected
end-to-end because example files are committed smoke samples.

```bash
# 1. start servers (leave running)
gym env start \
    --resources-server moldetox \
    --model-type openai_model \
    --model "<model id>" \
    --model-url "<openai-compatible base url>" \
    --model-api-key "$API_KEY"

# 2. collect rollouts against them
gym eval run --no-serve \
    --agent moldetox_simple_agent \
    --input resources_servers/moldetox/data/example.jsonl \
    --output resources_servers/moldetox/data/example_rollouts.jsonl
```

`gym env status` lists the three servers and their ports.

Prepare a full split — any of the eight configs, `test` only:

```bash
python resources_servers/moldetox/scripts/prepare_moldetox.py \
    --config task3_smiles_gen_single \
    --output resources_servers/moldetox/data/val.jsonl
```

Regenerate the committed example artifacts:

```bash
python resources_servers/moldetox/scripts/prepare_moldetox.py \
    --output resources_servers/moldetox/data/example.jsonl --limit 5
gym dataset collate \
    "+config_paths=[resources_servers/moldetox/configs/moldetox.yaml]" \
    +output_dirpath=resources_servers/moldetox/data \
    +mode=example_validation
```

## Tests

```bash
ng_test +entrypoint=resources_servers/moldetox
```

## Licensing

Code: Apache 2.0.

MolDeTox data: Creative Commons Attribution 4.0 International, declared on the
dataset card and ungated. Upstream it is derived from Therapeutics Data Commons,
FDA, and SIDER; see the [dataset card](https://huggingface.co/datasets/MolDeTox/MolDeTox)
for their terms.

**The system prompts in `prompts.py` are neither.** They are quoted verbatim
from Tables H, I, J and K of [arXiv:2605.12181](https://arxiv.org/abs/2605.12181),
and copyright in that text is held by its authors. The paper is distributed
under the [arXiv perpetual non-exclusive license](http://arxiv.org/licenses/nonexclusive-distrib/1.0/);
the dataset's CC-BY-4.0 covers the dataset, not the paper. Their inclusion was
reviewed and approved for this contribution on 2026-09-16.

`prompts.py` carries a third-party notice naming the authors and the source.
Keep it there if the file is moved or copied: the Apache-2.0 header covers the
NVIDIA-authored code around the constants, not the quoted text itself.
