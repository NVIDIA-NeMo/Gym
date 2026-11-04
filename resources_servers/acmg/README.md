# acmg resources server

Deterministic verifier for ACMG/AMP variant classification answers, used by the [`acmg` benchmark](../../benchmarks/acmg/README.md).

## Scoring

`official_reward.py` is the `nvidia/acmg_variant_benchmark` dataset's standalone `training/reward.py`, vendored byte
for byte from revision `59fb412e7e39e8b209f431fce522458587ca8fec` (upstream SHA-256
`e2a80a9d3d0cc5051d73c9d6ea6245e4e2272ea5ef00cb6375a978b152666f1f`). It is excluded from ruff in `pyproject.toml` and
`tests/test_scoring.py` checks the checksum of everything below the provenance header. Change `scoring.py`, not the
vendored file.

`verify()` takes the final assistant text (`response.output_text` with `<think>`/`<thinking>` blocks removed; nothing
is ever read from reasoning), parses the `CRITERIA:` and `CLASSIFICATION:` lines tolerantly (markdown, `PM2_Supporting`,
`PM2 (Supporting)`, full label names or a JSON answer), and returns:

- `reward`: the dataset's default scalar, `0.5 x label + 0.3 x criterion F1 + 0.2 x self-consistency`, 0.0 without a
  classification; when `label_scoring` is `excluded` the label drops out and the other weights are renormalised;
- `components`: `answered`, `label` (None when excluded), `criterion_precision`, `criterion_recall`, `criterion_f1`,
  `exact_criteria`, `self_consistent`, `evidence_distance`, `evidence_score`;
- `predicted_label`, `predicted_criteria`, and the pooled counts (`returned_n`, `returned_acceptable`, `required_n`,
  `required_hit`, `self_consistency_assessable`) that `compute_metrics` aggregates into `acmg/<arm>/<metric>`.

The two arms (`with_rules`, `without_rules`) are reported separately and never pooled. Self-consistency uses the rule
document's combining rules (`reference.combining`) in the with-rules arm and any standard ACMG/AMP system otherwise.

## Task data

`verifier_metadata` follows `task_data.TaskData`: `record_id` (`<case_id>:<arm>`), `case_id`, `arm`, `reference`
(`label`, `label_scoring`, `accepted_labels`, `required`, `acceptable`, `combining`) and case metadata (`gene`,
`role`, `skill`, `consequence`, `approved_2026`, `version_status`, `confirmation`, `rule_document`).
`data/example.jsonl` holds six synthetic rows (three invented cases, both arms) with a short synthetic system prompt
and rule document; it contains no benchmark cases, prompts or rule documents, which are canary-marked and private.

## Validation

```bash
gym env test --resources-server acmg
gym dataset collate --config resources_servers/acmg/configs/acmg.yaml \
    --output-dir /tmp/acmg-example --mode example_validation
```

The tests cover the verifier fixture, the vendored-scorer checksum, exact and partial answers, excluded labels,
reasoning removal, output failures, pooled per-arm metrics and the committed examples.

## Licensing

Gym integration code: Apache-2.0. The vendored scorer and the dataset are NVIDIA-authored; the dataset card declares
`license: other`, and the dataset is private. Dependency: `nemo-gym` (Apache-2.0).
