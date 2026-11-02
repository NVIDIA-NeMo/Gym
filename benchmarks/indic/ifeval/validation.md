<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# IndicIFEval validation

Validated on 2026-10-11 after merging Gym main at `4a9236e4`.
This records integration checks, not a benchmark quality baseline.

## PR #3695 follow-up validation (2026-10-14)

The [review request](https://github.com/NVIDIA-NeMo/Gym/pull/3695#discussion_r4243058513)
identified that instruction accuracies were present in `agent_metrics` but absent
from `key_metrics`. The Indic backend now selects the four overall strict/loose
prompt/instruction accuracies. The English backend retains its default `mean/*`
selection. A regression through `verify()` and `aggregate_metrics()` checks an
answer that satisfies one of two strict constraints: prompt strict is 0 and
instruction strict is 0.5, while both loose accuracies are 1.

The [core CI failure](https://github.com/NVIDIA-NeMo/Gym/actions/runs/38171481834/job/114565012701)
was caused by undeclared Indic fields in the instruction-following task schema:
`language`, `uuid`, `subset_for_metrics`, and `metadata`. These fields are now
declared without requiring them on English rows. Only `language` is marked as
read exclusively from `verifier_metadata`; the shared fields retain their
legacy placement compatibility, as established in
[PR #2827](https://github.com/NVIDIA-NeMo/Gym/pull/2827).

Both the missing-summary regression and the exact CI schema failure were
reproduced before the fixes. With Python 3.13.14, `PYTHONPATH=.`, and the existing
NLTK/checker caches, final targeted validation passed:

```bash
python -m pytest tests/unit_tests/test_indic_ifeval.py \
  tests/unit_tests/test_task_data.py tests/unit_tests/test_aggregate_metrics.py \
  -q --tb=short
python -m pytest resources_servers/instruction_following/tests -q --tb=short
pre-commit run --all-files
git diff --check
```

- **491 core tests passed**, including the full committed-data schema sweep,
  schema import isolation, preparation, and shared aggregation.
- **42 instruction-following tests passed**, including English compatibility
  and the new summary regression.
- **Pre-commit and diff whitespace checks passed.**
- Re-aggregated **24 saved real Gnani responses across 12 languages**, containing
  **48 instructions**, through the updated server. All four `key_metrics` equal
  independently calculated pooled accuracies and their `agent_metrics` values;
  all per-language metrics remain present. No checker errors were recorded.
  This is a reporting/schema change, so no new inference was needed; the model
  evidence below remains applicable to generation and scoring.

The previous PR head `e57aa625` passed all eight remote server shards. Its
separate [harness conformance failures](https://github.com/NVIDIA-NeMo/Gym/runs/114566215550)
also occur on [unchanged upstream main `4a9236e4`](https://github.com/NVIDIA-NeMo/Gym/runs/114616856089):
OpenCode reports two failed scenarios, Pi one, and Codex seven; Hermes passes.
Those unrelated adapters are outside this change. New-head CI still needs to
run; these local checks do not establish a green remote pipeline.

## Review feedback applied

The applicable feedback from [Gym PR #3596](https://github.com/NVIDIA-NeMo/Gym/pull/3596)
is covered as follows:

| Review concern | IndicIFEval implementation and check |
| --- | --- |
| Canonical split and repeat count | Manifest and config both use `benchmark`, one attempt per prompt, and matching dataset declarations. Static manifest validation passes. |
| Sampling lost when choosing a model server | Defaults are materialized on Indic rows. CLI-to-collector and vLLM conversion tests verify temperature 0 and 1,280 output tokens survive model selection. |
| Shared seeds and metadata | The merged upstream collector supplies independent metadata. Tests verify seeds 0–3, preserved `extra_body.top_k`, and unchanged source metadata when repeats are explicitly requested. |
| Defaults leaking into other benchmarks | Combined IndicIFEval/Numb3rs tests preserve audio metadata and routing; explicit CLI sampling overrides still apply across the run. |
| Shared English verifier disappearing | `_copy` keeps the original English resource instance. Two regression tests reproduced the missing-server error before the fix, in both benchmark orders. |
| Behavioral tests absent from CI | Preparation and composition tests live in `tests/unit_tests/test_indic_ifeval.py`. Scorer tests stay in the instruction-following server suite, and adapter checks stay in the vLLM server suite. |
| Server imports inflating core coverage | Preparation imports a small shared schema rather than the scorer or server implementation. A subprocess import check guards that boundary. |
| Current branch and honest evidence | Upstream main is merged; preparation, tests, saved-model replay, and live-smoke artifacts are recorded below. |

The dataset remains **unpinned**, as requested for this integration. The optional
fixed-dataset-revision suggestion is therefore not adopted. Preserve the prepared
JSONL with each run; its hash below identifies the tested content. The scorer
revision and archive checksum remain pinned independently of dataset selection.

## Preparation

With the existing evaluation Python environment active, `PYTHONPATH=.`, and
`HF_HOME` set to the campaign's existing Hugging Face cache:

```bash
HF_HUB_OFFLINE=1 python -m benchmarks.indic.ifeval.prepare \
  --output-path results/indic-ifeval-review/prepared.jsonl
gym env validate indic/ifeval
```

Production preparation read the real cached Parquet files, using the cached default
revision and the default `correct` quality filter: **5,731 prompts**. This was an
offline preparation check, not a claim that the remote dataset is unchanged.

| Language | bn | gu | hi | kn | mr | ml | ne | or | pa | ta | te | ur |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompts | 454 | 482 | 487 | 457 | 483 | 490 | 468 | 473 | 474 | 489 | 488 | 486 |

Prepared JSONL SHA-256:
`3a9af8e9aa7a8a4eae958f27f26911fdf24885bc00b74f2378270125ad73a668`.
Static validation also checked all 12 committed example rows.

## Automated checks

Python 3.13.14; NLTK resources and the pinned checker cache were pre-provisioned.
The existing campaign driver environment supplied the installed dependencies.

```bash
python -m pytest tests/unit_tests/test_indic_ifeval.py \
  tests/unit_tests/test_rollout_seed_metadata.py \
  tests/unit_tests/test_indic_aime_2026.py tests/unit_tests/test_benchmarks.py \
  tests/unit_tests/test_prompt.py -q --tb=short
python -m pytest resources_servers/instruction_following/tests \
  responses_api_models/vllm_model/tests/test_app.py -q --tb=short
python -m pytest tests/unit_tests/test_indic_ifeval.py \
  tests/unit_tests/test_rollout_seed_metadata.py -q --tb=short
python -m pytest tests/unit_tests/test_indic_ifeval.py \
  --cov=resources_servers.instruction_following.indicifeval_schema \
  --cov-report=term-missing --cov-fail-under=100 -q --tb=short
pre-commit run --all-files
git diff --check
```

- Core/config/preparation checks: **182 passed**, including benchmark discovery,
  prompt handling, and the Indic AIME regressions. Focused IndicIFEval/seed checks:
  **25 passed**.
- Scorer and vLLM checks: **293 passed, one pre-existing failure**:
  `TestEndpointFile::test_unpublished_endpoint_grace_lifecycle`. The same failure
  was reproduced using the unchanged test from upstream `4a9236e4` and its unchanged
  vLLM implementation: the test exceeds the 300-second unpublished-endpoint grace
  period without expecting the resulting exception. This is outside the Indic diff.
- The shared preparation schema reached **100% statement coverage** (21 tests).
- The existing instruction-following example-data validator passed (five examples,
  matching example metrics, five saved rollouts, and no conflict artifacts).
- Pre-commit and whitespace checks passed. CI has not been run for this local branch.

The expanded CI scope is the full suite because preparation tests now live under
`tests/unit_tests/`. The entire core/sandbox suite and eight component shards were
not run locally; the focused checks above are the recorded validation scope.

## Saved real-model replay

Replayed **24** existing `gnani/gnani-evon-v3.3-30B-A3B` smoke responses, two per
language, through the final instruction-following verifier. All **96** comparisons
of prompt/instruction strict/loose fields matched the saved results; zero checker
errors. These responses came from the thinking-on campaign with temperature 0.6,
top-p 0.95, top-k off, repetition penalty 1.0, and a 50,000-token output cap.
This replay checks scoring compatibility; it does not generate new model answers.

Local source artifacts: the sibling campaign directory
`../indic-ifeval-trans-50k-20260924/runs/gnani30_t06/<language>/validation/rollouts.jsonl`.
The replay report is `results/indic-ifeval-review/evidence-summary.json`; bulk
rollouts and local runner scripts are ignored by git.

## Fresh model smoke

Ran the native Gym collector with the final config and shared verifier against
an already-running `sarvam-external/sarvam105b-05102026` endpoint using the
`qwen3` reasoning parser. Selected real `correct` Hindi and Kannada source key
**102**, which requests exactly three Markdown bullets. Used temperature 0,
the benchmark's 1,280-token cap, concurrency 1, and an explicit smoke-only
`enable_thinking: false` override. The benchmark does not force this thinking
setting; it belongs to this smoke run.

**2/2 completed**, each with exactly three `* ` bullet lines, reward 1, strict
and loose instruction accuracy 1, and zero checker errors. Output lengths were
**128 tokens (Hindi)** and **152 tokens (Kannada)**. Inspected both final messages
and captured outgoing requests, confirming temperature 0, `max_tokens: 1280`,
and `chat_template_kwargs: {enable_thinking: false}`. No new GPU allocation was
needed. This is an integration smoke, not a quality or thinking-mode baseline.

An earlier thinking-on Hindi smoke at 8,192 output tokens exhausted the budget
before producing a final message. The response was marked incomplete and scored
zero for all three instructions, with no checker errors. The remaining long
smoke was stopped; the short completed-answer smoke above was used to inspect
normal final-message handling. This demonstrates why thinking runs need an
explicitly reported budget and why a reasoning-only reply must not earn reward.

Local artifacts under `results/indic-ifeval-review/` (ignored by git):

- `run-final-smoke.py`, `final-smoke-config.yaml`, `final-smoke-input.jsonl`, and
  `final-smoke-command.json`: the two-row selection, resolved local run setup,
  and exact command.
- `sarvam-final-rollouts.jsonl`, `sarvam-final-transport.jsonl`, and
  `live-smoke-summary.json`: complete outputs, captured requests/responses, and
  inspected results.
- `sarvam-smoke-rollouts.jsonl`: the earlier capped thinking-mode response.

Completed smoke JSONL SHA-256:
`f6434f03927984a8a8fae66db89b44c042085ac585b418f513ea3d02240b124c`.
Gym's general rollout-health report marked the two records **unobserved** because
this legacy collection did not emit agent-turn capture artifacts. The completion,
parser, request-setting, and verifier assertions above were checked directly;
they do not imply full token-capture health validation.
