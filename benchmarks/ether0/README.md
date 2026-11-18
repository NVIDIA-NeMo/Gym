# ether0

## Measurement contract

Source: [futurehouse/ether0-benchmark](https://huggingface.co/datasets/futurehouse/ether0-benchmark), revision `c7d5e59960087f360bc32a5006bb994324b38c35`, the `test` split ([paper](https://arxiv.org/pdf/2506.17238)).

Prepared task count: **325**, about 25 per task type across 14 types. Every answer is a molecule, a reaction or a reaction name. Data license: CC-BY-4.0. Integration code is Apache-2.0.

Preparation uses the environment's own formatter (`environments/ether0/prepare.py`), so prompts and `verifier_metadata` are the ones the [ether0 resources server](../../resources_servers/ether0/README.md) grades. The standard `simple_agent` is used with no tools.

Mean binary reward in [0, 1], higher is better. No LLM judge.

## Oracle files

Fifty tasks (25 `retro-synthesis`, 25 `oracle-solubility`) are graded by chemistry models that the resources server runs in-process. Missing MolTrans and the default ZINC20 catalog download on first use. For offline runs, supply existing files; see the [ether0 resources server README](../../resources_servers/ether0/README.md) for the files, their checksums and the alternatives.

## Prepare and run

```bash
gym eval prepare --benchmark ether0
gym eval run --benchmark ether0 --model-type vllm_model
```

Preparation writes `benchmarks/ether0/data/test.jsonl`, which is gitignored.
