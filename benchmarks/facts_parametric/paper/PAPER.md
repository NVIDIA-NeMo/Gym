# Benchmark technical report

- Title: The FACTS Leaderboard: A Comprehensive Benchmark for Large Language Model Factuality
- Authors: Aileen Cheng, Alon Jacovi, Amir Globerson, Ben Golan, Charles Kwong, Chris Alberti, Connie Tao, Eyal Ben-David, Gaurav Singh Tomar, Lukas Haas, Yonatan Bitton, et al. (Google DeepMind, Google Research, Kaggle)
- Canonical source: https://arxiv.org/abs/2512.10791v1 (version v1, 11 December 2025)
- Immutable PDF: https://arxiv.org/pdf/2512.10791v1
- SHA-256 of the PDF: `db046e76cc1877880843d0e7fd4898422f1064d47f8990b04f3c230229ede6be`
- Mirror published by the authors: https://storage.googleapis.com/deepmind-media/FACTS/FACTS_benchmark_suite_paper.pdf
  (SHA-256 `f079762c7a3ce79e725e79fa009fa3afd63eb3cbf77f6507c1f9901640752cb7`; same report, different build)
- License: CC BY 4.0
- Leaderboard: https://www.kaggle.com/benchmarks/google/facts-parametric
- Official public data: https://www.kaggle.com/datasets/kaggle/facts-parametric-public-examples (version 2, Apache 2.0)
- Official reference implementation: Kaggle starter notebook `yulongt/facts-parametric-benchmark-starter-code` (version 10)

The PDF is not committed to keep binaries out of the repository. `python benchmarks/facts_parametric/fetch_paper.py`
downloads the immutable arXiv version into this directory (gitignored) and verifies the SHA-256; run packages copy
the verified file under `paper/`. Section 4 (FACTS Parametric) and its Table 6 define the protocol this adapter
reconciles against: closed-book answers, three sampled Gemini 2.5 Pro grades per response, accuracy as the
primary metric, and hedging rate, attempted accuracy, and F1 as secondary metrics.
