# NOOA capability evaluations in Gym

This directory contains small NOOA agents used to exercise framework capabilities through Gym's full evaluation path. The first slice mirrors NOOA's `calculate_simple` capability fixture:

- source agent: `tests/capability/agents/calculate_single.py` in NVIDIA-NeMo/labs-OO-Agents;
- source cases: `tests/capability/data/calculate_simple.jsonl`;
- scoring: the dedicated `nooa_capability` Resources server, which preserves NOOA v0.0.9 `ExactMatchScorer` normalization and binary reward semantics.

The dataset copies the two source cases into a self-contained Gym JSONL file. The expected answer is verifier-only: the NOOA argument map exposes only `a`, `b`, and `calculation` to the generated method.

## Run

Start the composed environment with a configured Gym model server:

```bash
gym env start \
  --config responses_api_agents/nooa_agent/configs/nooa_calculate_capability.yaml \
  --resources-server nooa_capability \
  --model-type openai_model
```

Then evaluate both cases:

```bash
gym eval run --no-serve \
  --config responses_api_agents/nooa_agent/configs/nooa_calculate_capability.yaml \
  --resources-server nooa_capability \
  --model-type openai_model \
  --agent nooa_calculate_capability \
  --input responses_api_agents/nooa_agent/data/capability_calculate.jsonl \
  --output results/nooa-calculate-simple.jsonl
```

## Expansion order

After this exact-match smoke slice is stable, add capability families incrementally:

1. richer exact-match cases whose typed values survive JSON normalization;
2. structured extraction with parity-tested structured scorers;
3. `calculate_complex` and other trace-sensitive tests after native NOOA trace projection is available;
4. error recovery, stateful multi-turn, and router/subagent concurrency cases;
5. the truncation matrix and LLM-judge methodology scorers.

Trace-dependent and LLM-judge scorers should not be silently reduced to exact match. Add native NOOA trace projection first, then port their original scoring contracts.

## Hermetic process-level test

Run the full localhost topology (Gym CLI, NOOA agent server, OpenAI model proxy, deterministic Responses server, and capability Resources server) with:

```bash
bash tests/e2e/nooa_calculate_e2e_test.sh
```

The test targets Linux CI with Gym's pinned Python 3.13 and `uv` versions. By default it creates isolated component environments from the committed requirements, including the immutable merged NOOA commit that provides scoped hooks. `NOOA_SOURCE_DIR` and `SERVER_VENV` are development-only overrides for coordinated local testing.
