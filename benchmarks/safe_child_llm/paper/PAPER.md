# Benchmark paper

- Title: Safe-Child-LLM: A Developmental Benchmark for Evaluating LLM Safety in Child-LLM Interactions
- Authors: Rath, Junfeng Jiao and Saeid Ghasemi and Ryan Mcdaniel and others (The Responsible AI Initiative)
- Venue: arXiv preprint
- Canonical source: https://arxiv.org/abs/2506.13510v4 (version v4)
- Immutable PDF: https://arxiv.org/pdf/2506.13510v4
- SHA-256 of the PDF: `8d5402c24dbe9dc0ff9a818766c01ec93bb804d148a9e2b3f48c3796fdf48615`
- License: arXiv non-exclusive-distrib 1.0
- Pinned code/data revision this adapter reconciles against: `f69a651ff5c992c6d423b6a129ade8bf674fb63b`

The PDF is not committed. `python benchmarks/safe_child_llm/fetch_paper.py` downloads the immutable
version above into this directory (gitignored) and verifies the SHA-256; run packages copy the
verified file under `paper/`. The version pin is deliberate: later arXiv revisions may change
tables or protocol text, and the adapter was checked against v4.
