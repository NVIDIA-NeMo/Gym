<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# IndicIFEval-Trans scorer provenance

Source: https://github.com/AI4Bharat/IndicIFEval

- Commit: `1bb5f1bc4936cb20b8544e185bfd7cbed8f31464`.
- Archive: `https://codeload.github.com/AI4Bharat/IndicIFEval/zip/1bb5f1bc4936cb20b8544e185bfd7cbed8f31464`.
- SHA-256: `01692685e707e039a48b60680ea1436811325a1907f79c69ffb5ce2fd000ba9e`.
- Source directory: `lm-evaluation-harness/custom_configs/indicifeval-trans/`.
- Imported files: `utils.py`, `instructions_registry.py`, and the
  `*_instructions.py` / `*_instructions_util.py` pairs for
  `bn, gu, hi, kn, mr, ml, ne, or, pa, ta, te, ur`.

`setup_indicifeval.py` downloads the archive, verifies its checksum, and
extracts only those files plus the upstream MIT `LICENSE`. Google Research's
Apache-2.0 headers remain intact. The cache is ignored by git. Setup is
locked and publishes a complete directory atomically.

Packaging adaptations are deliberately limited to package-relative imports,
removing `sys.path.append` calls, and removing the full-registry debug print
in strict evaluation. Scoring functions and checker bodies are unchanged.
Modules load under a revision-specific namespace and cannot overwrite the
English `verifiable_instructions` registry.

The Gym adapter strips completed reasoning blocks, calls upstream strict and
loose evaluation for each instruction, records checker exceptions as explicit
failures, and aggregates the four metrics. It does not import the Ground
pipeline or add a general language requirement beyond the source constraints.

## Differences from the reference evaluation

- The reference repository itself is unchanged. Import adaptations apply only
  to Gym's generated cache; the upstream instruction rules are retained.
- Dataset preparation selects the 12 supported Trans language splits and,
  by default, rows tagged `correct`. The upstream task YAML does not apply
  this quality filter; its README recommends it for paper-aligned evaluation.
- Gym removes completed reasoning blocks before scoring. The upstream scorer
  receives its response string directly.
- Gym reports checker exceptions as failed instructions with error details;
  the upstream scorer lets those exceptions propagate.
- Gym exposes binary/fractional reward and aggregate metrics through its
  existing IFEval server. The English backend remains the default.

Source comparison confirmed identical non-import logic in all 24 selected
checker/utility files and all 21 evaluation functions (excluding the removed
debug print). A stratified comparison covered all 288 language/instruction
pairs present in the default dataset: 690 response evaluations and 2,760
metric comparisons, with no mismatches. These checks use fixed sample
responses; a live-model smoke evaluation is still pending.
