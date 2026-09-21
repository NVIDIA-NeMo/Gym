# NMRArena benchmark environment

[Repository and dataset](https://github.com/odanchem/NMRArena) (MIT, default branch
`release`; `main` does not exist) and
[preprint](https://doi.org/10.26434/chemrxiv.15006195/v1) ("Can AI Help Chemists
To Solve NMR?", Kliuev, Smirnov, Losev, Afanasyev and Chusov, ChemRxiv 2026).

Organic structure elucidation from NMR. The model reads the experimental 1H and
13C peak lists of one molecule (text, not images; no molecular formula) and
returns up to ten candidate structures as SMILES, ranked best first, in a JSON
object. A molecule is solved when a candidate is the same structure as the truth
after RDKit canonicalisation with stereochemistry stripped. 105 molecules, five in
each of 21 functional-group classes, curated from the OdanChem spectral database.

Upstream has no tags, releases or DOI for the data, and both the spectra strings
and the results table have been edited in place, so everything is pinned:

| Artifact | Pin |
| --- | --- |
| `odanchem/NMRArena` commit (head of `release`, 2026-09-17) | `8b4ca8a8953185c00f0c4d7fa3c16c23aa616326` |
| `dataset/dataset_selected_clean_105.json` SHA-256 | `6c8aed1adeecfb80278d47b4b524de54246b2739f9c568e3f2f2c7d7f7259108` |
| RDKit | `2026.03.3` (`requirements.txt`; upstream pins none) |

## Scope

The 105-molecule benchmark that every system in upstream's table is scored on.
Upstream's 1,015-molecule leakage-free set (`dataset_selected_clean_1015.json`)
is not prepared: the headline table is not reported on it, and only two
specialist systems were run against it.

Every verify response carries a `status`. A zero reward that is not a judgement
on the policy also carries a `failure_reason` and `harness_failure: 1.0`:

| Status | Meaning | Harness fault |
| --- | --- | :---: |
| `scored` | At least one candidate parsed and was compared with the truth | no |
| `empty_output` | No assistant text | no |
| `format_fail` | No loadable `"candidates"` JSON in the text (upstream's name) | no |
| `no_valid_candidate` | JSON found, but no entry parses as a molecule | no |
| `invalid_candidate` | `strict_candidates` only: an invalid, oversize, duplicate or eleventh entry | no |
| `bad_metadata` | The row's gold SMILES is missing, mistyped or does not parse | yes |

## Prompting

Both turns are upstream's, zero-shot. `prompts/system_prompt.txt` is the
`_SYSTEM_TEMPLATE` of `dataset/llm_track.ipynb` byte for byte (MIT, see
`prompts/NOTICE`), formatted with `n=10` and `min_slots=10` exactly as upstream
does; `prompting.py` reproduces the notebook's user prompt and its normalisation
of the peak strings (label dropped, `DMSO_d6` → `DMSO-d6`, `CDCl_3` → `CDCl3`,
whitespace squeezed). The prepared messages were checked equal to the requests
upstream logged in `results/LLM_results/llm_rep1_raw.jsonl`.

Each prepared row carries upstream's decoding: `temperature: 1.0` and
`max_output_tokens: 24576` ("identical settings for every model — temperature
1.0, 24K max tokens, provider-default reasoning"). Upstream sets no `top_p` and no
seed; neither is sent. Nothing about reasoning is sent either, so the provider's
default applies, as it did upstream. Upstream routed every model through
OpenRouter; a run through another gateway is protocol-matched in prompt and
decoding but not in provider, and should be described as a comparison.

The 24K budget covers reasoning as well as the answer. In upstream's own Gemini
3.1 Pro logs 14 to 20 of 105 calls per run stop at the cap with no complete JSON,
and the mean completion is about 22K tokens. Some gateways report such a call as
`status: completed` with empty text rather than `incomplete`; read
`mean/output_tokens` and `mean/response_incomplete` together.

## Dataset format

`scripts/prepare_nmrarena.py` downloads the pinned file, verifies its digest, and
fails closed unless exactly 105 records load in 21 classes of 5 with unique
`compound_id` values, non-empty spectra and a parseable gold SMILES for every row.
Each row carries:

- `responses_create_params.input` — the system prompt, then the user prompt;
  `temperature` and `max_output_tokens` as above
- `verifier_metadata.smiles` — the gold structure, read by `verify()`
- `verifier_metadata.compound_id`, `primary_class`, `n_complex` — echoed on the
  response for per-class and per-complexity breakdowns
- `verifier_metadata.publication_id`, `doi`, `dataset_commit`, `dataset_sha256`
  — provenance, never read
- `agent_ref` — `{"type": "responses_api_agents", "name": "nmrarena_simple_agent"}`

## Scoring

The candidate list is extracted as upstream's notebook does: the object around
the *last* `"candidates"` key in the text is brace-matched and loaded (comments
and trailing commas stripped on a second attempt) and read in `rank` order.
Upstream's published per-item lists additionally contain, for outputs cut off at
the token budget, the complete `{"rank", "smiles"}` objects written before the
cut, which the notebook parser cannot produce; `salvage_truncated_json: true`
(default) reproduces that recovery, `false` reproduces the notebook.

Scoring follows upstream's analysis notebook, which takes the published lists as
they are: the entry at position *i* has rank *i*, an entry RDKit cannot parse is a
miss at its position, duplicates occupy positions, and the first
`num_candidates` (10) positions are considered. `reward` is **Top-1**: the
position-1 candidate equals the canonical truth. Additional fields:

| Field | Meaning |
| --- | --- |
| `top1`, `top10` | 1.0 when the truth is at position 1 / within positions 1–10 |
| `hit_rank` | position of the first match, or `null` |
| `tanimoto_top1` | Morgan radius-2 (ECFP4), 2048-bit Tanimoto between the position-1 candidate and the truth; `null` when position 1 does not parse, so `mean/tanimoto_top1` is **conditional on answering**, as upstream reports it |
| `answered` | 1.0 when at least one candidate parses |
| `candidates` | canonical SMILES per position (`null` where the entry did not parse) |
| `n_raw_candidates`, `n_valid`, `n_invalid`, `n_oversize`, `n_duplicate`, `salvaged` | how the list was read |
| `response_incomplete` | the model response reported stopping at `max_output_tokens` |
| `truth_canonical`, `compound_id`, `primary_class`, `n_complex` | echoed from the row |

Headline (`key_metrics`): `mean/reward`, `mean/top1`, `mean/top10`,
`mean/answered`, `mean/response_incomplete`, `mean/harness_failure` and token
means. The conditional Tanimoto is deliberately not a headline: a model that
answers only when confident is paid for its silence (upstream's own footnote
reports 0.89 for a model that answered 9 of 315 times). It stays in
`agent_metrics` as `mean/tanimoto_top1` and `tanimoto_top1/answered_only`, next
to `count/answered`, so it is read with its denominator.

### Departures from upstream

- **Positional scoring, not the notebook's filtered list.** The notebook's
  `parse_candidates` drops invalid and duplicate entries and keeps the first ten
  valid; the published table is computed on the raw lists. On every list upstream
  published the two rules agree on Top-1 and Top-10 and differ on Tanimoto in the
  third decimal; the published rule is used because it is what the numbers are,
  and because the filtered rule pays Top-1 for "garbage first, truth second".
- **Length cap.** Any model-controlled SMILES longer than `max_smiles_chars` (500;
  the longest gold is 92) is a miss at its position without reaching RDKit, whose
  canonical ranking overflows the C stack on inputs around 15,000 characters.
- **Strict mode.** `strict_candidates: true` disqualifies a prediction containing
  any invalid, oversize or duplicate entry, or more than ten entries. Off by
  default; it exists to measure upstream's tolerance, not to replace it.

## Harness validation

No model is involved. `scripts/validate_harness.py` drives `verify()` over the
whole prepared benchmark and writes a JSON report:

- **Gold as prediction** scores Top-1 = 1.0, Top-10 = 1.0, Tanimoto = 1.0 on
  105 of 105.
- **Negative controls over all 105 rows**: empty output, an unparseable SMILES,
  the user prompt echoed back, a constant methane or benzene answer, and the gold
  at position 11 all score Top-1 = Top-10 = 0. Garbage-then-gold is Top-10 but not
  Top-1 in lenient mode and nothing in strict mode; gold-then-garbage is Top-1 in
  lenient mode and nothing in strict mode; a 20,000-character SMILES is a status.
  A constant trivial answer scores a conditional Tanimoto below 0.05, and an echo
  or empty answer leaves it undefined, so that metric does not pay for doing
  nothing — but it does pay for answering selectively, which is why it is not a
  headline.
- **Upstream parity.** Upstream's published per-item lists for one model are
  re-scored through `verify()` and compared item by item with upstream's own
  formulas: identical hit rank on every list, identical Tanimoto denominator, and
  a three-run mean ± SD that reproduces the README table.

These controls cover the named failure classes; they are not proof over every
wrong output.

## Setup

```bash
cd resources_servers/nmrarena
uv venv --python 3.13
uv pip install -r requirements.txt
cd ../..
source resources_servers/nmrarena/.venv/bin/activate
```

**Every command below runs from the repository root with that environment
active.** Preparation needs network access to `raw.githubusercontent.com`.

## Quickstart

```bash
# prepare the benchmark (one 74 KB download)
python resources_servers/nmrarena/scripts/prepare_nmrarena.py \
    --output resources_servers/nmrarena/data/nmrarena_105.jsonl

# validate the verifier model-free (downloads upstream's 840 KB per-item results)
python resources_servers/nmrarena/scripts/validate_harness.py \
    --input resources_servers/nmrarena/data/nmrarena_105.jsonl \
    --upstream-model gemini --output results/nmrarena/validation.json

# 1. start servers (leave running); the model id is the policy under test
gym env start \
    --resources-server nmrarena \
    --model-type openai_model \
    --model "<model id>"

# 2. collect one sweep against them
gym eval run --no-serve \
    --agent nmrarena_simple_agent \
    --input resources_servers/nmrarena/data/nmrarena_105.jsonl \
    --output results/nmrarena/run1.jsonl
```

`openai_model` reads `policy_base_url` and `policy_api_key` from `env.yaml` at
the repository root. Upstream's protocol is three independent sweeps; run the
collection three times to separate output files and summarise them with
`scripts/summarize_runs.py`.

Regenerate the committed example artifacts. `example.jsonl` is five synthetic
rows (textbook molecules absent from the benchmark, with peak lists written for
this repository); the benchmark's own spectra are third-party content and are not
redistributed here.

```bash
python resources_servers/nmrarena/scripts/make_example_data.py
gym dataset collate \
    "+config_paths=[resources_servers/nmrarena/configs/nmrarena.yaml]" \
    +output_dirpath=resources_servers/nmrarena/data \
    +mode=example_validation
```

## Tests

```bash
gym env test +entrypoint=resources_servers/nmrarena
```

## Licensing

Code: Apache 2.0.

NMRArena data: MIT ("Copyright (c) 2026 OdanChem"), declared for the whole
repository including `dataset/dataset_selected_clean_105.json`. The spectra are
curated from OdanChem's open spectral database; rights in the underlying published
spectra are not determined here. The data is downloaded at run time and not
committed.

`prompts/system_prompt.txt` is upstream's, unmodified, under MIT with the upstream
licence and a provenance notice alongside. The Apache-2.0 header covers the
NVIDIA-authored code around it, not the file itself.
