# OCSR (USPTO-OCSR-5719)

Optical chemical structure recognition: the model sees one molecule drawing from a US patent and
writes its SMILES. Source: the frozen USPTO-OCSR-5719 benchmark released by Edison Scientific with
OCSRGlyph ([`EdisonScientific/glyph-datasets`](https://huggingface.co/datasets/EdisonScientific/glyph-datasets),
`ocsr/uspto_ocsr_benchmark.parquet`; paper arXiv:2607.28532).

5,711 tasks: every RDKit-scoreable row of the 5,719.

## Provenance

- **Download, not re-host.** `prepare.py` fetches the parquet from the Hub at revision
  `86f8e17d75699d8ed0799eb6d100ceba3faac56e` and refuses to continue unless its sha256 is
  `b24e61e4e49c08b051d66a24a9ef8beafc6980c620bc5c976166e676c00fa52f` (the value in Edison's
  `uspto_ocsr_benchmark.parquet.provenance.json`). Nothing from the benchmark is committed here.
- **Images** are the public-domain USPTO patent depictions of
  [`hheiden/USPTO_OCSR_benchmark`](https://huggingface.co/datasets/hheiden/USPTO_OCSR_benchmark),
  embedded verbatim by Edison (all 5,719 are PNG). They go into the prompt unchanged, as a base64 data URL.
- **Labels**: 5,704 from `hheiden/USPTO_OCSR_benchmark`, 13 recovered by Edison from MolScribe's
  `real/USPTO.csv`, 2 empty (`source_of_label` in `verifier_metadata`).
- **Excluded (8 rows)**: the 2 empty labels and 6 labels RDKit cannot parse (valence or
  aromaticity errors in the label itself). Listed in
  `prepare.py:EXCLUDED_IDS`. All 5,711 kept labels parse under RDKit 2025.09.1 and 2026.3.6.
- **Licences**: `glyph-datasets` is Apache-2.0; the images are USPTO public domain;
  `hheiden/USPTO_OCSR_benchmark` carries no licence tag. Eval use only; Edison states the set was never
  used to train OCSRGlyph. Do not train on it.

## Task contract

- System: `You are an expert chemist specializing in optical chemical structure recognition (OCSR).`
- User: the instruction text (asks for `{"smiles": "<SMILES>", "confidence": <0-1>}` only) followed
  by the image.
- One turn, no tools. `max_output_tokens` 65,536 (reasoning models use 10k-30k tokens on hard drawings).
- The gold SMILES is only in `verifier_metadata`, never in the prompt.

## Scoring (`resources_servers/ocsr`)

- Answer = the last `{"smiles": ...}` object in the final message (reasoning removed); otherwise the last
  RDKit-valid SMILES in backticks (`format_ok` = 0 then).
- **Reward = `exact`**: RDKit canonical isomeric SMILES of prediction and gold are equal (stereo must be right).
- Diagnostics: `graph` (equal without stereo), `inchikey` (standard InChIKey equal), `tanimoto`
  (Morgan r=2, 2048 bits), `valid`, `format_ok`.
- A row with an unusable gold label is masked (`mask_sample`, `failure_kind: ocsr:bad_gold`), never scored 0.

## Grading, failures and time contract

- **Grading path: the resources server observes it (path A).** The score is a pure RDKit calculation with no
  model call, run inside `observe_programmatic_grading`, so every scored row, a wrong-answer zero included, carries
  a programmatic grading record bound to the graded response and reward. `REVERIFY_MODE` is stateless: grading a
  saved answer again gives the same reward.
- **Failures:**
  - A missing, unparseable or wrong answer is a plain, unmasked 0. So is an answer cut off by `max_output_tokens`.
  - A gold label RDKit cannot read is masked as `ocsr:bad_gold` and records no grade. The prepared 5,711 rows
    contain none (the unusable labels are excluded at preparation); the check guards other data.
  - Model and transport failures (for example `model_context_limit`, provider errors) are typed by Gym's model
    server and agent, not by this server. There is no judge, sandbox or tool.
- **Time contract:** no episode deadline. An episode is one model call with no tools, bounded by
  `max_output_tokens` (65,536) and by the model server's own request timeout. Verification is synchronous and
  takes milliseconds per row.
- **Resume identity:** rows are keyed by `verifier_metadata.id`. Data, repeats, model, prompt, sampling and
  grading are material settings.

## Reference numbers

Published by Edison for OCSRGlyph (arXiv:2607.28532): canonical SMILES 93.8%, graph 96.2%, over all
5,719 rows with their own metric definitions. This benchmark scores the 5,711 scoreable rows, so treat
those numbers as a calibration point, not a target. A reference run of this config on one frontier
model is still to be done (Gym exit check).

## Commands

```bash
gym eval prepare --benchmark ocsr                      # 5,711 rows, ~32 MB
gym env validate ocsr
gym env test ocsr && gym env test --resources-server ocsr

gym eval run --benchmark ocsr --model-type openai_model \
    --split benchmark --limit 5 --num-repeats 1 --output results/ocsr_smoke.jsonl
```

## Limitations

- One drawing style family (USPTO patent images, 2008 grants). Specialists trained on USPTO-680K
  score in-distribution here; general VLM scores may be inflated by memorised public molecules.
- `exact` requires correct stereo; read `graph` for connectivity only.
