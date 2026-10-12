# AssayBench

[AssayBench](https://github.com/Genentech/AssayBench) ([arXiv:2605.10876](https://arxiv.org/abs/2605.10876),
De Brouwer, Edwards et al., Genentech, May 2026) is a phenotypic-screen prediction benchmark built from 1,920
human CRISPR screens in BioGRID ORCS. Each screen is described in plain text (cell line, perturbation modality,
phenotype, significance criterion, treatment) and the model must return the 100 genes most likely to be hits,
ranked from strongest to weakest. The published headline is that zero-shot frontier LLMs lead (Gemini 3 Pro
0.157 AnDCG@100 on the test split), beat biology-specific models and trained predictors, and still sit far
below the empirical ceiling (Oracle kNN 0.292, technical replicates roughly double the best LLM).

- Task type: single-turn gene ranking
- Domain: `knowledge`
- Reward: **AnDCG@100** in [0, 1], continuous (0 is random-or-worse, 1 the ideal ranking)
- Data: [`Genentech/assaybench`](https://huggingface.co/datasets/Genentech/assaybench) on Hugging Face (MIT)

## Task format

Each row is one screen. `question` is the paper's Appendix A.4 template rendered for that screen; the verifier
reads `relevance_genes` (every gene assayed, HGNC) and `relevance_scores` (positive percentile-based relevance for
hits, 0 for non-hits, negative for hits in the opposite direction of a decomposed bidirectional screen). The model
answers with a comma-separated list of HGNC symbols.

The prompt is applied at run time from `benchmarks/prompts/eval/assaybench/paper.yaml` — see **Prompt** below.

## Verification

Everything that decides a score is the authors' own code, imported from the `assaybench` PyPI package
(pinned to 0.2.0 in `requirements.txt`) rather than reimplemented:

1. The gene list is read out of the reply the way the reference harness read it (**Prompt**, below), then
   split on commas by the harness's `parse_genes_from_output` — vendored verbatim in `gene_parsing.py`, since
   that script is not part of the package.
2. `assaybench.benchmark.metrics.RankingMetrics.evaluate(predicted, relevance_genes, relevance_scores)` with
   the package defaults: HGNC gene mapper on, `k_values=[5, 10, 20, 50, 100]`, thresholded scoring. This is
   the exact call the paper's results cache was built with (`figures/results_cache_data.py` upstream).
3. `reward = adjusted_ndcg@100`. Every other scalar the package returns rides along in `metrics`.

What AnDCG does, in one paragraph (paper §3.1, Appendix B.2): predictions are de-duplicated and canonicalized
to HGNC; a predicted gene not assayed in the screen is *unscored* and dropped after the top-k cut (a model cannot
know the library, so out-of-screen picks are not false positives); the list is zero-padded to k, so a short
list is penalized; DCG uses the signed relevance, so opposite-direction hits near the top cost score; nDCG is
normalized by the ideal ranking of the clipped-positive relevance; and the result is rescaled against the
screen's analytic random baseline, `AnDCG = max((nDCG - nDCG_rand) / (1 - nDCG_rand), 0)`. The test
`test_reward_is_the_paper_formula` writes this out independently and checks the package against it.

The response carries diagnostics that the aggregate metrics roll up:

| Field | Meaning |
|---|---|
| `extraction_mode` | `dspy_answer` (the `[[ ## answer ## ]]` field), `raw_fallback` (DSPy could not parse the reply; whole-reply scan), `quality_gate_fallback`, `none` |
| `dspy_parse_failed` | the reply did not follow the DSPy format — see **What does not line up** |
| `truncated` | the model hit its output budget (`incomplete` / `max_output_tokens`); the partial list is still scored, as upstream |
| `predicted_genes`, `num_predicted` | the parsed list before HGNC canonicalization |
| `metrics.hallucination_rate` | fraction of predicted symbols the mapper could not resolve to any HGNC gene |

```yaml
k_values: [5, 10, 20, 50, 100]      # RankingMetrics cutoffs
reward_metric: adjusted_ndcg@100    # which metric is `reward`
use_gene_mapper: true               # canonicalize predictions to HGNC (on in the paper)
min_expected_genes: 0               # upstream's quality gate; see app.py, off by default
strip_think_blocks: true            # drop a closed <think> block before parsing
```

## Metrics

`pass@1[avg-of-k]/adjusted_ndcg@100` is the paper's protocol — per-screen mean over the k runs, then mean over
screens — times 100, as every Gym aggregate is. So a paper value of 0.1211 reads as 12.11 here. Reported
alongside, at the same cutoff, are the other columns of the paper's tables: `precision@100`, `fdr@100`
(directional FDR), `normalized_precision@100`, `normalized_fdr@100`, plus `hallucination_rate`,
`dspy_parse_failed`, `truncated` and `empty_prediction` rates. All of it is also broken out by coarse phenotype
class (`fitness_proliferation_viability/...`, `drug_chemical_environmental_response/...`, …), which is the
paper's Figure 3.

Two properties are inherited from upstream and worth knowing before reading a number:

- `precision@k` / `fdr@k` exist only for rollouts with **at least k distinct predicted genes** (the package
  skips the key otherwise). A rollout without the key drops out of that metric's mean, exactly as in the
  reference aggregation, so a model that habitually returns 80 genes has a `precision@100` computed over the
  subset of rollouts where it returned 100. `adjusted_ndcg@k` is always present.
- On LaTest no screen has negative relevance, so `fdr@100` is 0 for every rollout there (the paper prints NA).

## Data

The paper reports three cohorts of the same task, all from year fold 0 of the pinned dataset revision. They
share one row format and one benchmark; the cohort is chosen at prepare time (default `test`):

| `+prepare_script_args.split=` | Cohort | Rows | Paper |
|---|---|---|---|
| `test` (default) | temporal **test** (published after 2021) | 334 | Table 2; Table 3 "test" |
| `validation` | temporal **validation** (published in 2021) | 218 | Table 3 "val" |
| `LaTest` | **LaTest**: screens from publications after Sept 2025, absent from BioGRID | 19 | Table 3 "LaTest"; §5.5 memorization probe |

Each prepare call overwrites `benchmarks/assaybench/data/assaybench_benchmark.jsonl`, so give each cohort its
own `--output` when collecting rollouts. Every row carries its cohort in `split`.

`benchmarks/assaybench/prepare.py` reads the parquet files straight from the Hugging Face snapshot (revision pinned in
`HF_REVISION`) with single-threaded `pyarrow`, because `datasets.load_dataset` memory-maps the 880 MB file and
fails under the address-space limits of a typical cluster login node. It refuses to write a cohort whose row
count differs from Table 1. Rows are ~170 KB each (the gene lists), so the JSONLs are gitignored and
regenerated: 56 MB for test, 32 MB for validation.

The rendered `question` and the two relevance lists were checked equal, row for row, against what upstream's
`AssayBenchDataset.get_list_examples` produces for the same 334 test screens.

The paper's 1,349-screen training split (the data behind its SFT and GRPO runs) is not declared here; it
would be a `train` dataset in the GitLab registry, and `build_rows("train")` already produces it.

The committed 5-row `data/example.jsonl` is **synthetic**: five invented screens over a 40-gene library that
exercise every prompt field and every relevance pattern (positive-only, positive + negative, few hits). It is
pre-materialized with the paper prompt, as every paired server's example fixture is, so do not pass
`+prompt_config` when rolling it out. `python create_examples.py --rollouts` regenerates it together with
`example_rollouts.jsonl`.

## Prompt

The messages the reference harness sent are three layers deep, and `benchmarks/prompts/eval/assaybench/paper.yaml`
reproduces all three:

1. **The paper's template** (Appendix A.4; `biogrid_ranking_prompt` in the package) rendered per screen. This is
   the row's `question`. `tests/test_prepare.py` checks the transcription against the installed package, and the
   trailing period of `phenotype` is dropped before rendering exactly as upstream's loader does.
2. **The collection-time suffix** the harness appended to every question
   (`collect_llm_predictions.py::process_single_example`): *"Your goal is to provide a list of genes that meet the
   screen criteria, even if you do not have access to the actual experimental data. … Do not refuse to answer …"*.
   It is not in the paper; it is in the code the paper's numbers came from.
3. **DSPy's `ChatAdapter` framing.** The harness wrapped every model in `dspy.ChainOfThought(RankingSignature)`,
   which adds a system message declaring the `question` / `reasoning` / `answer` fields and asks the model to
   answer under `[[ ## reasoning ## ]]`, `[[ ## answer ## ]]`, `[[ ## completed ## ]]` markers. The strings
   were captured from dspy 3.3.1 (`ChatAdapter().format(...)`); `TestPrompt` pins the rendered messages to that
   capture byte for byte. The verifier reads the reply back with the same rules as `ChatAdapter.parse` (first
   section per field wins, text before the first header ignored, both fields required).

The DSPy layer is not a stylistic detail: it is why replies are chain-of-thought followed by a marked answer
field, and why a plain "TP53, MYC, …" reply counts as a parse failure upstream.

## Reproducing the paper

### Table 3 — AnDCG@100 / Precision@100 / dFDR@100 per cohort

Zero-shot rows only (the SFT / GRPO / GEPA / few-shot / ensemble rows need training or extra machinery):

| Model | val | test | LaTest | Open weights |
|---|---|---|---|---|
| Gemini 3 Pro | 0.1716 / 0.3710 / 0.0208 | 0.1570 / 0.2226 / 0.0164 | 0.1113 / 0.2340 / NA | no |
| GPT-5.4 | 0.1658 / 0.3262 / 0.0162 | 0.1470 / 0.1980 / 0.0206 | 0.0982 / 0.2218 / NA | no |
| Gemini 3 Flash | 0.1556 / 0.2495 / 0.0260 | 0.1446 / 0.2009 / 0.0180 | 0.0908 / 0.2351 / NA | no |
| **GPT-OSS-120B** | 0.1268 / 0.2292 / 0.0164 | 0.1211 / 0.1757 / 0.0223 | 0.0826 / 0.2497 / NA | **yes** |
| Qwen3.5-2B | 0.0237 / 0.0695 / 0.0216 | 0.0284 / 0.0755 / 0.0216 | 0.0324 / 0.1342 / NA | **yes** |
| Gene-frequency baseline | 0.1691 / 0.4103 / 0.0606 | 0.1334 / 0.2204 / 0.0292 | 0.0888 / 0.1866 / NA | — |

The table's "Precision@100" is the package's `normalized_precision@100`: the paper's §3.2 formula divides by
`min(k, G+)`, the number of hits when a screen has fewer than 100, and that is what the published numbers match
(a GPT-OSS-120B rerun gave 0.184 normalized vs 0.119 raw against the printed 0.1757). "dFDR@100" is the raw
`fdr@100`. Read `pass@1[avg-of-5]/<metric>` and divide by 100.

### Measured with this server

Three models, three cohorts, the paper's protocol (5 runs per screen, per-screen mean then mean over screens).
AnDCG@100 / Precision@100 / dFDR@100; the paper's number in parentheses. `±` is the standard deviation of AnDCG
across the 5 runs.

| Model | test (334) | val (218) | LaTest (19) |
|---|---|---|---|
| GPT-OSS-120B, vLLM, 1 node | 0.1204 ±0.003 / 0.1839 / 0.0222 (0.1211 / 0.1757 / 0.0223) | 0.1243 / 0.2257 / 0.0167 (0.1268 / 0.2292 / 0.0164) | 0.0851 / 0.2111 / 0 (0.0826 / 0.2497 / NA) |
| DeepSeek-V3.2 thinking, vLLM, 2 nodes | 0.1127 ±0.002 / 0.1648 / 0.0226 (0.1076 / 0.1617 / 0.0211) | 0.1170 / 0.2022 / 0.0144 (0.1134 / 0.2065 / 0.0151) | 0.0773 / 0.2110 / 0 (NA) |
| Gemini 3 Flash, hosted API | 0.1524 ±0.002 / 0.1996 / 0.0185 (0.1446 / 0.2009 / 0.0180) | 0.1599 / 0.2788 / 0.0140 (0.1556 / 0.2495 / 0.0260) | 0.1078 / 0.2272 / 0 (0.0908 / 0.2351 / NA) |

The two open-weight rows land within 0.005 AnDCG of the published values on the test split, with the paper's
ordering (Gemini 3 Flash > GPT-OSS-120B > DeepSeek-V3.2) preserved. Gemini sits about 5% above its published
test number; the paper's proprietary rows are single-run, the `gemini-3-flash-preview` alias has moved since
the paper, and the thinking budget behind a hosted endpoint is not the paper's to specify -- none of which
this server controls. Over all 8,565 rollouts: 1 truncation, 0 empty replies, 4 DSPy parse failures, and a
hallucination rate below 1% for every model. DeepSeek's published rows come from the upstream leaderboard
(`docs/assets/data/leaderboard.json`), which prints every model the paper ran; Table 3 has only a subset.

Two operational notes from those runs:

- **Hosted endpoints with a per-request timeout** (the run above went through a LiteLLM gateway with a 360 s
  cap) will occasionally lose a long thinking reply. By default one failed `/run` ends the whole collection;
  pass `+route_failures_to_sidecar=true` so it is recorded in `rollouts_failures.jsonl` instead, and rerun
  with `--resume` to fill the gap. The gateway also does not return Gemini's reasoning as `reasoning_content`,
  so `mean/output_tokens` there includes it (about 12k per reply) while the scored text is the final answer.
- **Serve hybrid thinkers with their thinking flag and a reasoning parser** (DeepSeek-V3.2:
  `--tokenizer-mode deepseek_v32 --reasoning-parser deepseek_v3` on the vLLM side, and
  `chat_template_kwargs: {thinking: true}` on the `vllm_model` server config so every request carries it; the stock
  `vllm_model.yaml` sends no kwarg). Without the parser the trace lands in the reply; without the flag the model
  answers shorter and worse and every number still looks plausible. Check that the rollouts carry reasoning items
  before comparing.

### Sampling settings (upstream `benchmarking/configs/collect-*.yaml`)

The paper does not print them; they are in the reference harness's configs. Pass them to `gym eval run`:

| Model | temperature | max tokens | other | runs |
|---|---|---|---|---|
| GPT-OSS-120B (vLLM) | 1.0 | 16,000 | — | 5 |
| Qwen3-235B-A22B-Thinking-2507 (vLLM) | 0.6 | 81,920 | top_p 0.95, top_k 20, min_p 0 | 5 |
| DeepSeek-V3.2 (vLLM) | 1.0 | 128,000 | top_p 0.95, thinking on | 5 |
| Gemini 3 Pro / GPT-5.4 (API) | 1.0 | 32,000 | — | 1 |
| generic local default | 1.0 | 16,000 | — | 5 |

`num_repeats: 5` in the benchmark config is the open-weight protocol; the proprietary rows are single-run.
The harness's DSPy client sent one system and one user message and read `message.content` only, so serve
reasoning models behind a reasoning parser (`--reasoning-parser`) as upstream did, so the trace does not land in
the reply.

```bash
gym eval prepare --benchmark assaybench                                        # test split
gym eval run --benchmark assaybench --model-type vllm_model \
    --temperature 1.0 --max-output-tokens 16000 --output results/assaybench-test.jsonl
gym eval prepare --benchmark assaybench +prepare_script_args.split=validation   # then run again
gym eval prepare --benchmark assaybench +prepare_script_args.split=LaTest
```

### What lines up

- **Metric code is the authors'.** Same package, same defaults, same cutoffs; the results cache that fed every
  table and figure was built with `RankingMetrics(k_values=[5, 10, 20, 50, 100])` and nothing else.
- **Rows are the authors'.** Same Hugging Face revision, year fold 0, template rendered by the same rule;
  checked row for row on the test split.
- **Messages are the harness's**, all three layers, byte for byte.
- **Aggregation is the harness's**: per-screen mean over runs, then mean over screens; empty replies score 0
  and stay in the mean; a truncated reply is scored on what it managed to emit.

### What does not line up

- **No JSON-adapter retry.** When DSPy's `ChatAdapter` failed to parse a reply, DSPy silently re-issued the
  request through its `JSONAdapter` (a second model call with a JSON response format) and parsed that. A
  single-shot verifier cannot do this; a reply that DSPy would not parse goes straight to the harness's own
  last-resort whole-reply scan (`extract_genes_from_raw_response`), and `dspy_parse_failed` records how often
  it happened. For a model that follows the marker format this never fires; check the rate before comparing.
- **`retry_failed`.** The GPT-5.4 config re-ran screens whose runs all came back empty; the open-model configs
  did not. Gym keeps every empty rollout at 0.
- **The quality gate was dead code for API and vLLM models.** Upstream re-scanned the raw reply when the answer
  field held fewer than 20 genes, but the raw text it needed was only available from its Biomni client. It is
  therefore off here (`min_expected_genes: 0`); set 20 to turn it on.
- **Upstream's answer parser drops real symbols with lowercase letters** (`C1orf43`, `SEPTIN`-style renames are
  fine, `orf` genes are not) and never extracts a symbol from a noisy token, because its regex is anchored.
  This is reproduced, not fixed: it is part of the number.
- **DSPy version.** The harness's `dspy` pin is not published; the framing was captured from 3.3.1 and has been
  stable since 2.6, but an older `ChatAdapter` could differ in the trailing instruction sentence.

## Requirements

`assaybench==0.2.0` from PyPI (MIT), which brings `datasets`, `scipy`, `numpy`, `pyyaml`, `requests`,
`python-dotenv`. The HGNC tables ship inside the package; nothing is downloaded at verify time. Loading them
takes a few seconds at server start.

## License

- **Code**: Apache-2.0. `gene_parsing.py` vendors two functions from the upstream MIT-licensed harness (header
  in file; entry in `ATTRIBUTIONS.md`). The `assaybench` PyPI dependency is MIT.
- **Benchmark data**: `Genentech/assaybench` on Hugging Face is released under MIT. Its `biogrid` config is
  derived from [BioGRID ORCS](https://orcs.thebiogrid.org/), whose data is itself MIT-licensed; the 19 LaTest
  screens are derived from the supplementary data of five 2025-2026 publications, redistributed by the authors
  under the same dataset license.
- **Reference tables the verifier reads**, all bundled inside the `assaybench` wheel and used by its
  `GeneMapper`: the HGNC symbol/alias tables (HGNC data is CC0), a UniProt protein-to-gene mapping (UniProt
  is CC BY 4.0), and the authors' own manual alias list (MIT). Nothing is downloaded at verify time: the
  mapper's fallback download from genenames.org only runs when the bundled cache is missing, which it never is.
- **Not used**: the upstream package's `THIRD_PARTY_NOTICES.md` flags the DepMap Public 26Q1 common-essentials
  file (DepMap Terms of Use, research-only) and several MSigDB collections (mixed licenses). Those back the
  `%essential`, pathway-overlap and Effective-Pathways metrics of the follow-on AssayLoop paper
  (`assaybench.benchmark.sequential` / `diversity` / `effective_pathways`). This server calls only
  `assaybench.benchmark.metrics.RankingMetrics`, which does not read them; none of those files is downloaded,
  bundled, or referenced here, and the package never fetches them without an explicit user action.
