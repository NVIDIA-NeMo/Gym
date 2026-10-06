# vera_mh

VERA-MH recommended profile (arXiv:2605.13318; https://github.com/SpringCare/VERA-MH at `2c9d1fc`, rubric v1.2): 100
clinician-authored personas x 2 user simulators (GPT 5.2, Claude Opus 4.5), one 30-turn conversation each, judged by
GPT 5.4 at `reasoning_effort=low`, pooled.

- Environment, endpoints and run instructions: [`resources_servers/vera_mh/README.md`](../../resources_servers/vera_mh/README.md)
- Metrics, provenance, upstream mapping, divergences and reading guide: [`METRICS.md`](METRICS.md)

```bash
gym eval prepare --benchmark vera_mh          # downloads the pinned personas.tsv, verifies every hash, writes 200 rows
shasum -a 256 benchmarks/vera_mh/data/vera_mh_recommended.jsonl
gym eval run --benchmark vera_mh --model-type vllm_model --model <id> --model-url <base>/v1 --model-api-key $KEY
```

`prepare.py` is deterministic: it refuses to run if the persona sheet or any vendored rubric file differs from the
pinned SHA-256, so the row hash only changes when upstream moves.

## Calibration against upstream

`calibrate.py` runs the pinned upstream code, in its own checkout and environment, over a finished run:

```bash
git clone https://github.com/SpringCare/VERA-MH.git /path/to/vera-mh && git -C /path/to/vera-mh checkout 2c9d1fcbb68e1a2df64171c18b3e4d4c18b2f89e
(cd /path/to/vera-mh && uv sync --frozen --no-dev)
python -m benchmarks.vera_mh.calibrate --rollouts results/vera_mh/<model>.jsonl --out results/vera_mh/calibration/<model> \
    --upstream-repo /path/to/vera-mh --upstream-python /path/to/vera-mh/.venv/bin/python
```

- `materialization`: persona prompts, transcripts and message lists recomputed by upstream, compared per conversation;
- `replay`: the judge's answer sequence fed to upstream's `LLMJudge` through a mock LLM; dimension ratings and
  `yes_question_id`s must agree exactly;
- `score`: the ratings written as an upstream `results.csv` and scored by `judge.score.score_results`; pooled and
  per-dimension scores must agree;
- `live` (`--live-size N`, needs judge access through `OPENAI_BASE_URL`/`OPENAI_API_KEY`): upstream `judge.py` re-judges
  a stratified subset with the same judge settings; per-dimension agreement measures judge stability, not the port.
