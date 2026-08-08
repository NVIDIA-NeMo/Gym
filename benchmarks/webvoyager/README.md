# WebVoyager

The first Gym profile runs the 643 official WebVoyager tasks through
`browsergym/openended`. It preserves the upstream action surface (`Click`,
`Type`, `Scroll`, `Wait`, `GoBack`, `Google`, `ANSWER`) with a safe translation
to BrowserGym high-level calls. The final answer and latest screenshots are
scored by the separate WebVoyager VLM judge.

The default source is `../WebVoyager/data/WebVoyager_data.jsonl`. Set
`WEBVOYAGER_SOURCE_JSONL` for another checkout and run:

```bash
gym eval prepare --benchmark webvoyager
```

The benchmark targets live public sites, so results are time-sensitive and
less reproducible than the self-hosted Arena benchmarks. Configure a
vision-capable judge through `webvoyager_judge_*` values in `env.yaml`, or
replace the default judge model config with another server named
`webvoyager_judge_model`.

The benchmark profile sends one current SoM screenshot plus a compact list of
labelled interactive elements to the policy, and replaces older visual
observations with an omission marker. The judge independently retains the
latest three screenshots. This matches upstream WebVoyager's context shape
without changing WebArena or VisualWebArena defaults.

For the committed ArXiv smoke row, load configs in this order. The private file
contains `policy_base_url`, `policy_api_key`, and `policy_model_name`; never
commit it. Using the policy as judge is only appropriate for integration smoke.

```bash
PINNED_GYM=/path/to/locked/gym
PRIVATE_CONFIG=/path/to/private/inferencehub-env.yaml

"$PINNED_GYM" eval run \
  --config benchmarks/webvoyager/config.yaml \
  --config "$PRIVATE_CONFIG" \
  --config benchmarks/webvoyager/configs/inferencehub_same_model.yaml \
  --config benchmarks/webvoyager/configs/arxiv13_smoke.yaml \
  --model-type openai_model \
  --split benchmark \
  --output /path/to/run/rollouts.jsonl \
  --limit 1 \
  --num-repeats 1 \
  --concurrency 1 \
  --temperature 1.0 \
  --max-output-tokens 1000
```

Use a locked CLI environment. A fresh root-level `uv run gym` can resolve a
different Ray version from the component environments and fail before task
execution; such a run is an infrastructure failure, not a zero reward.

The original Selenium runtime is intentionally not included in this first
version. It can be added later behind the same common protocol without changing
the dataset or agent contract.
