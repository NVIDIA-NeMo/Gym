# Agent Security Bench (ASB) — NeMo Gym adapter

[ASB](https://github.com/agiresearch/ASB) (ICLR 2025, [arXiv:2410.02644](https://arxiv.org/abs/2410.02644))
benchmarks prompt-injection, memory-poisoning and backdoor attacks against tool-using LLM
agents across ten professional scenarios.

Pinned upstream revision: `1f561dccf92d55302368fa67679b4ba9d9c8fdc4`.

## The thing to understand first

**ASB does not ship a dataset.** It ships the tools that construct one. The benchmark
comes into existence only when the runner crosses agents x tasks x attacker tools and
assembles the injected prompts from f-strings at execution time. There is no file in the
upstream repository you can point an evaluation at.

That has two consequences this adapter is built around:

1. The expansion has to be reproduced exactly, including which subset of tasks the
   published runs actually used — see **Denominators** below, and `METRICS.md` for the
   derivation. Getting this wrong silently produces a benchmark that is not ASB.
2. Because the rows are generated rather than distributed, they are **materialized once
   and pinned**, not regenerated per run.

## Denominators: 400 rows per condition

`main_attacker.py` defaults `--task_num` to 1 and the DPI/OPI/MP/mixed driver never
overrides it, so each published condition is `10 agents x 1 task x 40 agent-matched
attacker tools = 400 rows`. `config/POT.yml` sets `task_num: 2` over a 5-agent file:
`5 x 2 x 40 = 400` as well.

This is confirmed three independent ways — the published percentages quantize to 1/400,
upstream's shipped memory stores hold ~400 records each, and the expansion lands on 400
exactly. Full detail in [`METRICS.md`](METRICS.md).

**27 conditions x 400 = 10,800 rollouts per model.**

Expanding the full 51-task list (2,040 rows/condition) gives a benchmark 5x larger whose
numbers match no published cell. It is available via `--task-num` and is not ASB.

## Layout

| Path | What it is |
|---|---|
| `benchmarks/asb/upstream_spec.py` | Pinned revision, input hashes, and every load-bearing upstream string |
| `benchmarks/asb/prepare.py` | Fetch, materialize, verify, and the Hugging Face round trip |
| `benchmarks/asb/reporting/report.py` | Renders the published table shape |
| `resources_servers/asb/app.py` | ASR, utility, refusal judge, memory retrieval |
| `responses_api_agents/asb_agent/app.py` | The plan-then-execute loop from `ReactAgentAttack` |
| `benchmarks/asb/run_all_models.sh` | Per-model campaign runner |

Upstream is cloned into `benchmarks/asb/upstream/ASB` and **never vendored** — that path
is gitignored, and `prepare.py` verifies every input against a pinned SHA-256 before use.

## Usage

```bash
# Fetch upstream at the pinned revision and expand the matrix (writes 27 JSONL + manifest)
python -m benchmarks.asb.prepare materialize

# Or restore the pinned rows from the Hub instead — what a scheduled run should do
python -m benchmarks.asb.prepare pull

# Confirm the pinned rows still match upstream (reporting signal; never auto-overwrites)
python -m benchmarks.asb.prepare verify

# Run the matrix against every configured model
bash benchmarks/asb/run_all_models.sh

# One model, on its own port block (models can run in parallel this way)
ENV_YAML=env.ultra.yaml HEAD_PORT=11410 PORT_LOW=20001 PORT_HIGH=21500 \
  bash benchmarks/asb/run_all_models.sh ultra

# Render the published tables
python -m benchmarks.asb.reporting.report results/asb/*.jsonl --out benchmarks/asb/reports/asb.md
```

### Ports

Gym defaults to head port 11000 and the 10001–20000 range. This host runs several Gym
stacks at once, so the runner pins a head port and port block per invocation. A root
`env.yaml` is auto-loaded by Gym and will **shadow those settings** — the runner writes
`env.<key>.yaml` instead and passes it explicitly, and both patterns are gitignored.

## Storing rows on the Hugging Face Hub

The rows live in a dataset repo (default `snorkel-fdr/asb-selectors`) and the local copy is
gitignored. **Only inputs are published.** Rollouts and scores are per-checkpoint outputs
and stay in `results/`.

The expansion is deterministic and takes seconds, so this is a **pin, not a cache**. It is
worth doing because:

* Upstream is a live repository, and one of its data files is *already* a fetch that failed
  open into HTML (`agent_task_pot_all.jsonl`). A weekly job that re-clones inherits that
  class of silent corruption.
* The paraphrase-style defenses rewrite the task with an auxiliary LLM before the model
  under test sees it, so regenerating changes the *input* and breaks week-over-week
  attribution.
* A scheduled evaluation should not depend on github.com.

`prepare verify` re-expands from upstream and diffs against the pinned content hash. Run it
alongside the weekly job as a **non-blocking report**, so a genuine upstream fix surfaces
instead of being hidden by the pin.

## Deviations

Five, all forced by the environment rather than chosen, all disclosed in
[`METRICS.md`](METRICS.md) and rendered into every report. METRICS.md is canonical:

1. **Output cap 256 → 4,096 tokens.** Upstream's cap predates reasoning models, which spend
   it entirely on the trace and emit no plan — driving ASR to zero for non-security reasons.
2. **Memory ranking.** The poisoned corpus is upstream's own, extracted from its shipped
   Chroma stores; the ranking over it is lexical rather than `text-embedding-ada-002`.
3. **Plan-parse salvage.** Upstream's strict `json.loads` is tried first; markdown-fence and
   embedded-array salvage follow, and the path used is recorded per row.
4. **Consecutive system messages merged.** ASB sends two; Qwen3.5's template rejects that
   with a 400. Merged for every model so all four answer identical input.
5. **Tool schema envelope.** ASB declares `parameters: None` (and omits the key entirely on
   the attacker tool); the Responses API requires a schema object, so an empty one is
   supplied. Behaviourally inert -- ASB never passes tool arguments -- and the tool list
   rendered into the planning prompt keeps upstream's original text.

A sixth entry, the refusal judge's non-random row loss, is a measurement limitation rather
than a protocol change we made; see METRICS.md.

## Licensing

ASB is Apache-2.0 upstream. Nothing from it is vendored here; the checkout is pinned and
gitignored, and generated rows are published separately on the Hub.
