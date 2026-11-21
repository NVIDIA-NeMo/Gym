# Integration validation — 2026-11-18

## Dataset publication

- Uploaded to `anushakamathofficial/indic-arena-proxy-v3`, commit
  `da63b1db1770bc219962b196f4f93b066fa2e400`.
- Verified remote size and SHA-256 for all 26 source JSONL files (153,976,021 bytes).
- 5,200 prompt records and 5,200 matching frozen reference records; 400 each per
  language. The existing target repository was public and remains public.
- The upload contains only the 26 data files, dataset card, checksums, and the
  repository's existing `.gitattributes`. Logs, partial files, local keys, and
  candidate generations were excluded.
- Existing HF authentication was verified as `anushakamathofficial`; cached
  active and stored-token files had permissions `0600`. No token was printed or
  added to this checkout.

## Preparation and protocol

Full preparation from the supplied local source passed pinned file checksums,
question/reference ID joins, exact conversation/message-digest equality, reference
status/model checks, language checks, and cross-language question-ID checks.
Output contains 5,199 rows: 400 per language except Tamil, which has 399.
The Tamil failed translation `019c83f8-0d3e-7703-9603-03d6478253e1` remains in the
source dataset and is listed in the preparation report's exclusions.

A direct Hub preparation smoke downloaded the pinned English and Tamil source
and reference files, verified their hashes, validated all 400 rows per language,
and produced three eligible rows per language. This was a data preparation
smoke, not model inference.

Both parsed judge prompt templates match the supplied `chat400_arena*.yaml`
templates exactly. They were composed from the existing in-repository Arena
templates with the conversation-aware wording. The shared judge's thinking
filter is opt-in; existing benchmarks retain their previous default.

## Automated checks

119 tests passed across:

```text
tests/unit_tests/benchmarks/test_indic_arena_proxy_v3.py          38
tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_metrics.py  28
tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_publish.py   5
resources_servers/arena_judge/tests/test_app.py                  48
```

Coverage includes full conversation preservation, failed translations, stale
references, ID collisions, checksum mismatches, pinned Hub downloads, CPU-only
configuration, swapped verdicts, strong-win weighting, deterministic bootstrap,
missing/failed rows, stale scoring metadata, task envelopes, English separation,
smoke-headline suppression, credential signature rejection, and upload allowlists.
The existing scikit-learn scorer emits its upstream `penalty` deprecation warning;
this does not affect these test outcomes.

Checks used Python 3.13.14 from the existing GPQA environment with this checkout
on `PYTHONPATH`. Current `openai==2.45.0`, `aiohttp==3.14.4`, and this checkout's
package metadata were isolated under `/tmp/indic-arena-proxy-v3-deps` where
needed. No shared environment was modified.

Ruff lint and formatting checks and `git diff --check` passed for the changed
files. Shell syntax checks passed for all four workflow scripts. A mocked
submission/generation/judging flow checked `afterok`, failure exit propagation,
language validation, credential ownership/permissions, removal of token values
from submission exports, output paths, summary interpreter selection, and
context/output-budget validation. This mock did not submit Slurm jobs.

Pre-commit checks passed on all 24 staged integration files before committing.
The private guardword hook skipped its scan because its private catalog was
unavailable. A separate staged-file scan found no HF tokens, NVIDIA API tokens,
or private-key signatures, and confirmed that generated datasets were excluded.

The Gym parser validated generation and reverification configurations; the CPU
configuration contains only a judge model server and resources server. The
benchmark scanner identifies exactly one benchmark configuration. The runtime
dataset overlay uses `.yml` so it is not discovered as another benchmark.
`gym env validate indic/arena_proxy_v3` passed manifest, component, and dataset
validation, reporting all 5,199 prepared rows. Catalog discovery also printed an
existing unrelated `opencode_sandboxed_search_agent` migration warning.

## Remaining external validation

No GPU generation or live NVIDIA judge request was performed. A representative
real smoke still needs a checkpoint, an authorized Slurm account, and the user's
NVIDIA judge-key file. Endpoint/model availability and score agreement with a
reference run remain unverified. The benchmark stays experimental and
`verified: false`. No PR or full repository test run was made.

## Score comparison follow-up (November 20, 2026)

The benchmark now extends the existing Gym vLLM model server through its request
and converter hooks. It uses the reference chat endpoint, plain-text messages,
sampling parameters, server-side token count, and per-prompt output budget
`min(120000, context_length - prompt_tokens - 64)`. It also preserves the reference
final-answer extraction before Gym conversion, including prefixes and repeated
or unfinished thinking blocks, and has no total policy-request timeout. The
reference thinking preflight runs before collection. Gym core, the agent,
environment server, judge prompts, and scoring implementation were not changed.

The preparer now applies smoke limits to the source prefix before translation
exclusions. The summary counts a successfully reverified saved answer despite its
historical failure sidecar, while retaining Gym's authoritative managed-outcome
selection and rejecting inconsistent saved-answer identities. Optional
`--reference-verdicts` compares complete saved judgment populations using the
same prepared-input order for both bootstraps, with artifact hashes, verdict
differences, and a guarded 12-language Indic average/delta. English remains separate.

181 focused tests passed, with clean exits:

```text
tests/unit_tests/benchmarks/test_indic_arena_proxy_v3.py             39
tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_generation.py  31
tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_metrics.py     58
tests/unit_tests/benchmarks/test_indic_arena_proxy_v3_publish.py      5
resources_servers/arena_judge/tests/test_app.py                     48
```

These checks used Python 3.13.14 with `-S` and explicit dependency paths to avoid
the adjacent GPQA environment's editable-package imports. Module-path assertions
confirmed the tests loaded this checkout's code. The generation tests exercise
the full Gym Responses-to-chat model path, not just the budgeting helper.

Offline differential harnesses executed the supplied source functions directly:
42 generation cases matched tokenize/chat payloads, budgets, final answers, and
length status across conversation types, context sizes, and thinking-tag cases;
14 judge cases matched request bodies and accepted text/verdicts across both
templates, including empty, invalid, and truncated replies. Six additional
sanitization/thinking cases matched. These harnesses did not make model calls.

Ruff, formatting, shell syntax, diff checks, and scoped pre-commit hooks passed.
The private guardword catalog remained unavailable, so that hook skipped its
content scan; a separate scan of the staged files checked credential signatures
and confirmed that downloaded data was not staged.

Existing Gym transport retry, cancellation, and schema-validation policies remain.
Use completed, valid-response reference runs for comparison; the standalone
generator can persist HTTP 200 responses with `finish_reason=abort`, which Gym
treats as an infrastructure failure. Historical reference CIs may differ solely
because they used asynchronous completion order. Matching the offline contract
does not establish identical stochastic generations or live hosted-judge results.

The existing cached Hugging Face login was verified as `anushakamathofficial`;
its credential directory is `0700`, and token files are `0600`, owned by the user.
The dataset is now private. An authenticated full Hub preparation verified all
26 pinned source files and produced 5,199 rows with SHA-256
`9d8d4544f4d2770b7f22fcfe73d2227321133062848a1c9b3d8318951b90dbcb`.
The downloaded cache and prepared data are owner-only. The code requests cached
authentication with `token=True`; no token value is embedded or printed.

A live paired evaluation still needs a checkpoint, Slurm account, and NVIDIA
judge-key file path. No GPU job or live judge request was submitted in this follow-up.
