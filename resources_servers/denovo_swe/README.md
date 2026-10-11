# DeNovoSWE resources server

Verification for [AweAI-Team/DeNovoSWE](https://huggingface.co/datasets/AweAI-Team/DeNovoSWE), a
document-to-repository training dataset: each image ships a package's source plus a written spec,
and the agent must regenerate the package from the spec alone.

Unlike the other SWE servers, `seed_session` wipes the pre-existing source
(`_denovoswe_clean.sh`) and re-injects the spec as `README.md` before the agent starts. There is no
golden `patch`: the image's pre-existing source is the golden answer.

Sandboxes come from `nemo_gym.sandbox`, so the same server runs on OpenSandbox or any other
configured provider.

## Grading

Verification wipes the source again, re-injects the spec, applies the candidate patch, lays down the
canonical test suite from `test_patch`, reinstalls the package, and grades with the per-file pytest
evaluator in `_denovoswe_eval.py` (see `verification.py`).

## Prepare the data

`prepare_denovo_swe.py` only writes rows on `data/supported_instance_ids.txt` (3,076 instances whose
golden source resolved in all three passes of a full-set golden-patch sweep).

```bash
DENOVO_SWE_LIMIT=200 python -m resources_servers.denovo_swe.prepare_denovo_swe
```

## Golden-patch validation

Grades each task with the image's own source, which measures the dataset rather than a model.

```bash
gym env start \
  --config resources_servers/denovo_swe/configs/denovo_swe.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

python resources_servers/denovo_swe/apply_golden_patch.py \
  +training_jsonl=resources_servers/denovo_swe/data/denovo_swe_training.jsonl \
  +output_jsonl=results/denovo_swe_golden_patch.jsonl \
  +concurrency=32 +limit=100
```

`aggregate_golden_patch.py` joins repeated passes into supported / flaky / broken / inconclusive
buckets, and `diagnose_failures.py` separates infra faults from row failures.

## Running an agent

`configs/denovo_swe_opencode.yaml` wires the opencode sandboxed agent to
`denovo_swe_resources_server`, pointed at `data/denovo_swe_training.jsonl`.

```bash
gym env start \
  --config resources_servers/denovo_swe/configs/denovo_swe_opencode.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml
```

The DeNovoSWE dataset is licensed CC BY 4.0. This adapter is Apache-2.0.
