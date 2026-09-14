# Benchmark paper

- Title: HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal
- Authors: Mazeika, Mantas and Phan, Long and Yin, Xuwang and Zou, Andy and Wang, Zifan and Mu, Norman and Sakhaee, Elham and Li, Nathaniel and Basart, Steven and Li, Bo and Forsyth, David and Hendrycks, Dan
- Venue: Proceedings of the 41st International Conference on Machine Learning (ICML 2024)
- Canonical source: https://arxiv.org/abs/2402.04249v2 (version v2)
- Immutable PDF: https://arxiv.org/pdf/2402.04249v2
- SHA-256 of the PDF: `e6cbcbedbef6aebea3b6fc6c8fd7933546b26cf0a1e65bcf128eea235007c732`
- License: CC BY 4.0
- Pinned code/data revision this adapter reconciles against: `8e1604d1171fe8a48d8febecd22f600e462bdcdd`

The PDF is not committed. `python benchmarks/harmbench/fetch_paper.py` downloads the immutable
version above into this directory (gitignored) and verifies the SHA-256; run packages copy the
verified file under `paper/`. The version pin is deliberate: later arXiv revisions may change
tables or protocol text, and the adapter was checked against v2.
