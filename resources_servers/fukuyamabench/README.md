# FukuyamaBench benchmark environment

Step-by-step organic reaction-mechanism prediction. Given atom-mapped starting
reactants and reaction conditions, the model predicts the full sequence of
elementary steps through to the final product. Scoring is deterministic RDKit
structure comparison against expert-curated mechanisms — there is no judge model
anywhere in the path.

- Paper: [arXiv 2607.12771](https://arxiv.org/abs/2607.12771), "Learning
  Mechanistic Reasoning for Chemical Reactions with Large Language Models"
- Upstream: [HaCTang/ReactionMechanismReasoning](https://github.com/HaCTang/ReactionMechanismReasoning)
- Pinned revision: `63bb79f912f0b2de593996b80ffeea894f6f1a59`

## Scope

Upstream ships 319 cases in three difficulty tiers, derived from *Fukuyama's
Advanced Organic Reaction Mechanism* (Kagaku-Dojin, 2005):

| Set | Cases | Gemini-3.0-Pro top-1 |
| --- | --- | --- |
| A | 78 | 22.3% |
| B | 131 | 7.8% |
| C | 110 | 1.0% |

`prepare_fukuyamabench.py` defaults to **sets B and C** (241 cases), which are
the tiers carrying the sub-20% published figures. Set A is available via
`--sets A` but is a materially easier task; scores should never be aggregated
across tiers.

Upstream defines three tasks. This implements **pathway & product prediction**,
the headline task (the paper's Table 2) and the only one upstream ships an
inference harness for: the model is given the starting state and conditions but
*not* the final product, and must infer the intermediates and the product
jointly.

## Prompting

The system prompt and user template are copied from upstream
`eval/prompts/infer_pathway_prompts.yaml` at the pinned revision. The user
template is byte-identical; the system prompt is content-identical with trailing
whitespace stripped by this repository's pre-commit hook. Starting
reactants are supplied atom-mapped, matching upstream's SFT/RL input format.

The paper's Appendix C prints a *longer* prompt than the repository ships — it
adds an output-guidelines block and a one-shot worked example. Neither this
implementation nor upstream's own harness is byte-identical to what produced the
published tables.

## Dataset format

One row per case. `verifier_metadata` carries the gold mechanism and its
checkpoints; see `task_data.py` for the full field documentation.

```json
{
  "responses_create_params": {"input": [{"role": "system", ...}, {"role": "user", ...}]},
  "verifier_metadata": {
    "case_id": "B001",
    "case_set": "B",
    "gt_pathway": [{"step_id": "1", "products": ["CNC(O)CCCC=O"]}],
    "checkpoints": [[1], [2], [3, 4]],
    "lenient": true,
    "n_gt_steps": 13
  },
  "agent_ref": {"type": "responses_api_agents", "name": "fukuyamabench_simple_agent"}
}
```

## Scoring

The model emits a JSON array of steps, each with a `product_smiles`. Scoring
walks the case's **checkpoints** in order:

- A checkpoint is a group of gold step IDs that are mutually acceptable at that
  point. Upstream builds them so that bond-topology changes and functional-group
  transformations are checkpoints, while consecutive trivial proton transfers are
  merged. There are therefore usually fewer checkpoints than mechanism steps.
- Products are compared as **canonical SMILES sets**, after atom mapping is
  stripped. Any valid writing of the same structure matches.
- The scan over predicted steps only ever moves **forward**, and a matched step
  is consumed. This lets a model predict finer-grained steps than the gold
  mechanism without penalty, which is the reason checkpoints exist.
- **Reward is all-or-nothing**: 1.0 only if every checkpoint matched. Because a
  failed checkpoint exhausts the cursor, one miss also fails every later
  checkpoint. `checkpoint_accuracy` is reported alongside as a partial-credit
  diagnostic but is not the reward.
- Under `lenient` (the default, matching upstream) a predicted product set that
  is a non-empty subset of the gold set counts as a match, crediting a model that
  names the main product but omits a leaving group. `--strict` at prepare time
  requires exact set equality.

### Two deliberate departures from upstream

**Gold products are split on `.` just as predictions are.** Upstream splits only
the prediction side, so a gold entry written as a single dotted string (e.g.
`C=C1...CO.[Pd]`) can never be matched by a model that writes the same species —
those checkpoints are unmatchable by construction. Feeding the gold pathway back
in as the prediction fails 11 of the 241 set-B/C cases under upstream's rule and
5 under this one. Scores on affected cases will therefore be slightly *higher*
than upstream's scorer would report.

**Organometallic gold products are reparsed without valence checking.** RDKit's
valence model rejects ligand counts that are ordinary for Rh, Cr, Ti and Mo
centres, so those species fail the default parse — and an unparseable gold
product makes its checkpoint unmatchable by any prediction. Under upstream's
scorer this silently costs five set-B/C cases (`B042`, `B127`, `C069`, `C081`,
`C093`), capping the attainable score at 236/241. `canonical_smiles` therefore
retries failures with everything except `SANITIZE_PROPERTIES`, which keeps ring
and aromaticity perception intact and drops only the valence assertion.

The fallback is a retry, not the default, so the 1,716 species that parse
strictly are canonicalised exactly as before. It recovers 17 organometallic
species, and the distinct-canonical-form count rises by exactly 17 — every
recovered species maps to its own form, and no previously-separate species are
merged. Gold-as-prediction goes from 236/241 to **241/241** with all negative
controls still at zero.

### Metrics are reported per tier

`compute_metrics` emits `B/...` and `C/...` keys via `compute_subset_metrics`
keyed on `case_set`. The default Gym aggregation would report only a pooled
`mean/reward`, and since the published B and C baselines differ by close to an
order of magnitude, a mean over a mixed split describes no benchmark anyone
reports. Read the per-tier keys; treat the pooled mean as an artifact.

### Reproducing published pass@k

Upstream reports pass@k for k in {1, 3, 5, 8} over n=8 samples per case, using
the unbiased estimator. This server scores a single rollout, so pass@k is an
aggregation concern for the harness, not the verifier. Note also that upstream's
inference script runs the *first* of its n draws at temperature 0.0 and the rest
at 0.7 — pass@1 there is not a mean over n stochastic draws. The sampling
parameters behind the published tables are not stated in the paper.

## Quickstart

The example file needs servers started separately; `gym eval run` only accepts
`--input` together with `--no-serve`.

```bash
# 1. start servers (leave running)
gym env start \
    --resources-server fukuyamabench \
    --model-type openai_model \
    --model "<model id>" \
    --model-url "<openai-compatible base url>" \
    --model-api-key "$API_KEY"

# 2. collect rollouts against them
gym eval run --no-serve \
    --agent fukuyamabench_simple_agent \
    --input resources_servers/fukuyamabench/data/example.jsonl \
    --output resources_servers/fukuyamabench/data/example_rollouts.jsonl
```

Prepare a full split (downloads the pinned upstream tarball; no Hugging Face
dataset exists for this benchmark):

```bash
python scripts/prepare_fukuyamabench.py --output data/val.jsonl
python scripts/prepare_fukuyamabench.py --output data/val_c.jsonl --sets C
```

Regenerate the committed example:

```bash
python scripts/prepare_fukuyamabench.py \
    --output data/example.jsonl \
    --sets S --source-dir tests/fixtures/synthetic_mechanisms
```

### The committed example is synthetic

`data/example.jsonl` is built from five hand-authored mechanisms in
`tests/fixtures/synthetic_mechanisms/`, not from the benchmark — see Licensing
below. They are classic textbook-independent reactions (aldol addition, SN2, E1,
acid-catalysed hydration, ester saponification) emitted in the upstream
directory layout and passed through the same `prepare.py` path, so a row is
identical in shape to a real one.

The set deliberately covers the shapes that matter: a single-step case, a case
with fewer checkpoints than steps, a checkpoint grouping two equivalent steps,
and a gold product written as one dotted string. Feeding their gold pathways
back in scores 5/5, and all four negative controls score 0.

**This does not affect evaluation results.** Real runs use `val.jsonl`, which
`prepare.py` generates from the pinned upstream download and which is never
committed.

## Harness validation

The model-free check is **gold-as-prediction**: feed each case's gold pathway
back in and confirm it scores 1.0. Over sets B and C this passes **241/241**.
Negative controls (empty pathway, wrong-but-valid chemistry, unparseable SMILES,
and the gold pathway reversed) all score 0 on all 241 cases.

The two directions matter together: gold-as-prediction shows the scorer never
rejects a correct answer, and the negative controls show it never accepts a
wrong one. Either alone is passable by a scorer that always returns the same
value.

## Tests

```bash
ng_test +entrypoint=resources_servers/fukuyamabench
```

## Dependencies

The scorer needs a chemical-graph parser and a canonical-SMILES writer; the
standard library has neither, so one third-party toolkit is required.

**RDKit is selected.** It is the toolkit upstream's own scorer uses, so parity is
a property of the choice rather than something to be argued. It is BSD-3-Clause
(compatible with this repository's Apache-2.0 contribution rule), ships CPython
3.13 wheels, is on conda-forge, and its only direct Python dependencies are NumPy
and Pillow. It reproduced every control recorded above.

**Open Babel 3.2.1** is the strongest alternative and does support canonical
SMILES on Python 3.13, but it is **GPL-2.0**, which CONTRIBUTING forbids
introducing into the main tree. That rules it out independently of its merits.

Version: see the note in `requirements.txt` for why `2025.9.6` is pinned rather
than the newer `2026.3.6`, and what would justify moving it. Assessed
2026-09-14 against
[RDKit releases](https://github.com/rdkit/rdkit/releases),
[RDKit license](https://github.com/rdkit/rdkit/blob/master/license.txt), and
[Open Babel license](https://github.com/openbabel/openbabel/blob/master/COPYING).

## Licensing

- **Code**: Apache 2.0. `metrics.py` is a port of upstream's Apache-2.0 scorer;
  the prompt constants in `scripts/prepare_fukuyamabench.py` are reproduced from
  upstream's Apache-2.0 prompt file. Modifications are noted at their site.
- **Committed data**: none from the benchmark. `data/example.jsonl` is generated
  from the synthetic fixtures described above.
- **Benchmark data**: downloaded at runtime by `prepare.py` from upstream; not
  redistributed here.

**Unresolved:** upstream declares its dataset Apache-2.0, but the mechanisms are
transcribed from a commercial textbook (*Fukuyama Mechanism Book*, Kagaku-Dojin,
2005, ISBN 9784759810455) and neither the paper nor the repository states that
permission was obtained — upstream's `LICENSE` is the unmodified template with
the copyright holder never filled in. Since Apache-2.0 conveys only the rights
the licensor holds, that grant may not cover the derived content. Shipping no
benchmark rows keeps this repository clear of the question; clear it with Legal
before redistributing any real case.
