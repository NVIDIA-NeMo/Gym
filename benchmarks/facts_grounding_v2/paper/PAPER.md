# Benchmark technical reports

## FACTS Grounding v2 (the protocol this adapter implements)

- Title: The FACTS Leaderboard: A Comprehensive Benchmark for Large Language Model Factuality
- Authors: Aileen Cheng, Alon Jacovi, Amir Globerson, Ben Golan, Charles Kwong, Chris Alberti, Connie Tao, Eyal Ben-David, Gaurav Singh Tomar, Lukas Haas, Yonatan Bitton, et al. (Google DeepMind, Google Research, Kaggle)
- Canonical source: https://arxiv.org/abs/2512.10791v1 (version v1, 11 December 2025)
- Immutable PDF: https://arxiv.org/pdf/2512.10791v1
- SHA-256 of the PDF: `db046e76cc1877880843d0e7fd4898422f1064d47f8990b04f3c230229ede6be`
- Mirror published by the authors: https://storage.googleapis.com/deepmind-media/FACTS/FACTS_benchmark_suite_paper.pdf
  (SHA-256 `f079762c7a3ce79e725e79fa009fa3afd63eb3cbf77f6507c1f9901640752cb7`)
- License: CC BY 4.0
- Leaderboard: https://www.kaggle.com/benchmarks/google/facts-grounding
- Official public data: https://www.kaggle.com/datasets/deepmind/FACTS-grounding-examples (version 17)
- Official reference implementation: Kaggle starter notebook `prathameshbang/facts-grounding-v2-benchmark-starter` (version 4)

## FACTS Grounding v1 (prompt collection and the original judge design)

- Title: The FACTS Grounding Leaderboard: Benchmarking LLMs' Ability to Ground Responses to Long-Form Input
- Authors: Alon Jacovi, Andrew Wang, Chris Alberti, Connie Tao, Jon Lipovetz, Kate Olszewska, Lukas Haas, et al.
- Canonical source: https://arxiv.org/abs/2501.03200
- Public data mirror: https://huggingface.co/datasets/google/FACTS-grounding-public (revision `11b6961370aa0ac73c91d58e35317e724b2ac765`, 860 rows)

The PDF is not committed to keep binaries out of the repository. `python benchmarks/facts_grounding_v2/fetch_paper.py`
downloads the immutable arXiv version of the suite report into this directory (gitignored) and verifies the
SHA-256; run packages copy the verified file under `paper/`. Section 6 of the suite report defines the v2 protocol:
unchanged prompts, Gemini 2.5 Flash and GPT-5 as judges with the v2 prompt, an unadjusted factuality score averaged
over judges, and disqualification of ineligible responses.
