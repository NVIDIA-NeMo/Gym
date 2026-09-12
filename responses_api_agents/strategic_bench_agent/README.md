# Strategic Bench Multi-Turn Agent

This responses API agent simulates a multi-turn adversarial negotiation between a `policy model` and an `opponent model` (the environment's human user simulation) built to wrap the `StrategicBench` negotiation environment into the NeMo Gym paradigm.

## Execution
Run the multi-turn agent using an ASGI server like Uvicorn or using the provided run scripts:
```bash
bash scripts/run.sh
```