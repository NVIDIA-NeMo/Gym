# OCRBench v2 (OCRBench_v2, via VLMEvalKitMcore)

OCR mega-benchmark (10k samples, 30+ task types, EN + ZH) over the shared
`vlm_eval_kit` resources server. The mcore class is the monolith
`OCRBench_v2` (`vlmeval/dataset/image_vqa.py:3513`, dataset name `OCRBench_v2`).

- **Resources server:** `resources_servers/vlm_eval_kit` (`_score_OCRBench_v2`)
- **VLMEvalKit source:** the config opts into the mcore fork via `vlmevalkit_url`/
  `vlmevalkit_commit` (dual-source pattern; server default stays upstream) — the
  verified numbers below were produced against that pin
- **Judge:** none — every metric is rule-based, exactly as in the reference
  `evaluate()`.
- **Scoring:** per-sample REUSE of the mcore dispatcher `process_predictions`
  (`vlmeval/dataset/utils/ocrbrnch_v2_eval.py:44` — note the `ocrbrnch` typo in the
  module name), called with a single-item list in a worker thread. It dispatches on
  `data_item['type']` (= the Gym `category`) to TEDS (table/chart parsing), KIE-F1,
  IoU/spotting, the BLEU/METEOR/F-measure/edit-distance family (full-page OCR,
  translation), counting, and VQA containment/ANLS. Spotting rows are serialized
  behind a server-level lock: the mcore `spotting_evaluation` uses fixed cwd-relative
  scratch dirs, so concurrent calls silently zero each other.
- **Continuous rewards (deviation from binary-reward guidance):** many types produce
  raw scores in [0, 1] (TEDS, KIE-F1, ANLS, BLEU-family) — the reference metric itself
  is continuous and the official aggregates average these raw scores, so the per-sample
  `reward` is the raw score, not a 0/1 threshold.
- **Aggregation:** the server's `compute_metrics` reuses the mcore
  `ocrbench_v2_aggregate_accuracy` (`ocrbrnch_v2_eval.py:360-441`) to bucket per-type
  scores into EN/ZH capability areas (emitted as `OCRBench_v2/<bucket>`), then emits
  the headline **`OCRBench_v2_EN` / `OCRBench_v2_ZH`** as the unweighted mean over the
  present buckets — exactly `OCRBench_v2.evaluate` (`image_vqa.py:3560-3565`). Rows the
  reference marks `ignore` (e.g. empty-GT translation rows) are skipped, as in the
  reference (`ocrbrnch_v2_eval.py:371-372`). NOTE: the sample-weighted
  `mean/OCRBench_v2` is NOT the official number.
- **Verified numbers (Nemotron-3-Nano-Omni-30B, reasoning-on, full 10K set):**
  EN 67.26 vs paper 67.0 — REPRODUCED (+0.26); ZH 54.09 vs paper 52.7 — NEAR (+1.39,
  D-002); gate PASS (1/10000 empty). The as-run cluster numbers were wordnet-poisoned
  (missing nltk corpus zeroed every METEOR-using row); the setup fix in this branch
  installs the corpora at setup time and fails setup loudly if they are unusable —
  the numbers above are the wordnet-corrected rescoring.
- **Data:** built by `prepare.py` (`prepare_OCRBench_v2` in the server's
  `prepare_data.py`) from the auto-downloaded `OCRBench_v2.tsv` (~2 GB, md5-checked).
  Rows carry everything the dispatcher reads: `category` (the reference
  `data_item['type']`), `question`, literal-parsed `answer`/`bbox`/`content`, and
  `eval` when present ('multiple choice' / 'case sensitive') — parsing mirrors
  `evaluate` (`image_vqa.py:3535-3557`).
- **num_repeats:** 1

## Targets and numbers

|Metric key|Paper target (Nano 3 Omni, arXiv 2604.24954, reasoning-on)|Gym number (wordnet-corrected)|
|---|---|---|
|`OCRBench_v2_EN` (bucket-averaged — the official EN headline)|**67.0**|**67.26** full 10K set — REPRODUCED (+0.26)|
|`OCRBench_v2_ZH` (bucket-averaged — the official ZH headline)|**52.7**|**54.09** full 10K set — NEAR (+1.39, D-002)|
|`mean/OCRBench_v2` (sample-weighted — NOT the official number)|n/a|internal signal only|

## Runtime dependency note

The scoring path imports the mcore metric stack at verify time: `apted`, `zss`,
`distance`, `editdistance`, `Levenshtein`, `lxml` (TEDS), `nltk` + `jieba` (BLEU/METEOR
for full-page OCR / translation types), `Polygon3` (spotting), and `ipdb` (imported
unconditionally by the mcore modules). All are declared in the server's
`pyproject.toml`. The METEOR metric additionally needs the NLTK `wordnet` corpus at
runtime — `setup_VLMEvalKit` installs `wordnet` + `omw-1.4` into the server venv's
`nltk_data` (offline-safe guard; `punkt` deliberately excluded — nothing in the scorer
path tokenizes), warms the lazy corpus loader in the server process, and fails setup
loudly if the corpus is unusable, so a missing corpus can never silently zero scores
again.

## Run

```bash
# Prepare data (creates the server venv + pinned mcore checkout on first run,
# downloads OCRBench_v2.tsv into the LMU data root)
ng_prepare_data "+config_paths=[benchmarks/ocrbench_v2/config.yaml]" +output_dirpath=data/ocrbench_v2

# Serve + collect (no judge needed)
ng_run "+config_paths=[benchmarks/ocrbench_v2/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
ng_collect_rollouts +agent_name=ocrbench_v2_benchmark_simple_agent \
    +input_jsonl_fpath=benchmarks/ocrbench_v2/data/ocrbench_v2_benchmark.jsonl \
    +output_jsonl_fpath=results/ocrbench_v2_rollouts.jsonl +num_repeats=1
```

## Parity check (definition of done)

Reconstruct the reference predictions file from rollouts (rows carry `index` and
`question`) and score with the official `OCRBench_v2.evaluate()`; the
`English/Chinese Overall Score` entries must match `OCRBench_v2_EN` / `OCRBench_v2_ZH`
(same per-sample scoring and bucketing code on both paths).

## Parallel work — scorer reconciliation needed before merge

Maciej Mikulski's public-Gym branch (`mmikulski/vlmeval-generic-audio` on
github.com/NVIDIA-NeMo/Gym) carries an independent OCRBench_v2 scorer rewrite
(claimed 15/15 parity vs the native dispatcher) plus a `scoring_backend`
(local|vlmeval) switch. This branch's `_score_OCRBench_v2` instead REUSES the
mcore `process_predictions` dispatcher per sample. The two approaches must be
reconciled before both merge — reviewers: see the MR description; the
architectural decision goes through Maciej.
