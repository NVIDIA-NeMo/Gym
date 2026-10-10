# FACTS Multimodal resources server

This directory contains the NeMo Gym resources server, separate coverage and
factuality judge prompts, and configuration for FACTS Multimodal.

`configs/reverify.yaml` starts only the resources server and judge model for
`gym eval reverify`; it re-scores stored policy responses without policy
inference.

The benchmark’s full documentation—including its evaluation protocol, image
materialization and preparation, configuration, limitations, and validation
placeholders—lives in
[benchmarks/facts_multimodal/README.md](../../benchmarks/facts_multimodal/README.md).
