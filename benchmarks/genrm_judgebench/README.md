# genrm_judgebench

Prepare [ScalerLab/JudgeBench](https://huggingface.co/datasets/ScalerLab/JudgeBench):

```bash
uv run python -m benchmarks.genrm_judgebench.prepare
```

The default `gpt` split has 350 response pairs (700 comparisons).
Use `--split claude` for 270 pairs (540 comparisons).
Each pair is evaluated in both response orders. Following upstream JudgeBench,
correct judgments count +1, incorrect judgments -1, and unparseable judgments 0.
A pair earns credit when the sum is positive. Incomplete pairs count as incorrect.
Source revision: `57dd5e0b9817d07f05ec8f45a91b2ce1e310e308`.

See [GenRM evaluation](../../resources_servers/genrm_eval/README.md) for evaluation commands and scoring.
