# genrm_rmbench

Prepare public [RM-Bench](https://github.com/THU-KEG/RM-Bench):

```bash
uv run python -m benchmarks.genrm_rmbench.prepare
```

1,327 prompts × 9 style combinations × 2 response orders = 23,886 comparisons.
Source commit: `73c52d7b27b361621361ec959ac6c0eb9bb7a689`.

See [GenRM evaluation](../../resources_servers/genrm_eval/README.md) for evaluation commands and scoring.
