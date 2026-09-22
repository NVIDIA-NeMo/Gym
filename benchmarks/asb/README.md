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

### You do not have to generate the data yourself

There are two supported ways to get the rows, and they produce the same 10,800 rows:

- **Use the frozen copy.** One expansion of the matrix is already public, under ASB's own
  MIT license, so you can reproduce a number here without re-deriving anything or hosting
  a copy first. `python -m benchmarks.asb.prepare pull` fetches it; no credentials, no
  configuration.
- **Regenerate from upstream.** `python -m benchmarks.asb.prepare materialize` clones the
  pinned revision, checks every load-bearing input by SHA-256 and runs the expansion
  itself. Slower, and it needs network access to GitHub, but it depends on nothing but
  upstream.

Either way, `python -m benchmarks.asb.prepare verify` re-derives the rows from upstream
and diffs the content hash, so the frozen copy is a convenience you can check rather than
a source you have to trust. Run it before trusting any number.

The frozen copy is
[`theverifier/asb-selectors`](https://huggingface.co/datasets/theverifier/asb-selectors)
(10,800 rows at upstream `1f561dccf92d`, content hash `6351e7ee6a94fd6a`). It is
third-party rather than NeMo-owned; point `ASB_HF_REPO` or `--repo-id` at your own mirror
if you would rather not depend on it.

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

Upstream is cloned into `benchmarks/asb/upstream/ASB` and **never vendored** — that path
is gitignored, and `prepare.py` verifies every input against a pinned SHA-256 before use.

## Usage

```bash
# Either regenerate the matrix from upstream (writes 27 JSONL + manifest) ...
python -m benchmarks.asb.prepare materialize

# ... or pull the frozen public copy of the same rows. No credentials needed, and what a
# scheduled run should do so that every run scores identical inputs.
python -m benchmarks.asb.prepare pull

# Confirm the pinned rows still match upstream (reporting signal; never auto-overwrites)
python -m benchmarks.asb.prepare verify

# Start the ASB resources server, agent, policy model, and refusal judge.
# Supply model-server configuration for policy_model and asb_judge_model.
gym env start \
  --config resources_servers/asb/configs/asb.yaml \
  --model-type inference_provider

# In a second terminal, run a smoke rollout against the committed examples.
gym eval run --no-serve \
  --agent asb_agent \
  --input resources_servers/asb/data/example.jsonl \
  --output results/asb-example.jsonl \
  --limit 3

# Render the published tables
python -m benchmarks.asb.reporting.report results/asb/*.jsonl --out benchmarks/asb/reports/asb.md

# Optional: graded diagnostics derived from the same rollouts, no re-run
python -m benchmarks.asb.reporting.graded results/asb/*.jsonl
```

`report` renders ASB's published shape and its reward stays binary, because that is the
benchmark's own definition. `graded` is separate and **not comparable to any published
number**: it reads the per-step tool calls the rollouts already carry and reports when in a
trajectory the attacker tool was first invoked, what fraction of the normal tools were
invoked, and refusal as three values rather than two -- including the rows whose judge reply
could not be parsed, which the headline rate has no way to show.

## Storing rows on the Hugging Face Hub

Hosting your own pinned copy: pass `--repo-id`, or set `ASB_HF_REPO`. **`push` has no
default** and will refuse to guess, because publishing into somebody else's namespace by
accident is worse than an argument error; `pull` does default, to the public frozen copy
described above. The local materialized copy is gitignored. **Only inputs are published** —
rollouts and scores are per-checkpoint outputs and stay in `results/`.

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

ASB is MIT-licensed upstream. Nothing from it is vendored here; the checkout is pinned and
gitignored, and generated rows are published separately on the Hub.
