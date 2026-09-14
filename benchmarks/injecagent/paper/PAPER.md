# Benchmark paper

- Title: InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents
- Authors: Zhan, Qiusi and Liang, Zhixiang and Ying, Zifan and Kang, Daniel
- Venue: Findings of the Association for Computational Linguistics: ACL 2024
- Canonical source: https://arxiv.org/abs/2403.02691v3 (version v3)
- Immutable PDF: https://arxiv.org/pdf/2403.02691v3
- SHA-256 of the PDF: `49e6e6a4a00c797a772ef905a154c0839426655027d5fa0ce6b632e6c1db2bfb`
- License: arXiv non-exclusive-distrib 1.0
- Pinned code/data revision this adapter reconciles against: `f19c9f2c79a41046eb13c03c51a24c567a8ffa07`

The PDF is not committed. `python benchmarks/injecagent/fetch_paper.py` downloads the immutable
version above into this directory (gitignored) and verifies the SHA-256; run packages copy the
verified file under `paper/`. The version pin is deliberate: later arXiv revisions may change
tables or protocol text, and the adapter was checked against v3.
