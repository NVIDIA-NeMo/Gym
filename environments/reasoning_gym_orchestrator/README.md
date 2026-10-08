# Environment moved

This compatibility directory preserves the former `reasoning_gym_orchestrator` entry point. Use [`langgraph_orchestrator_reasoning_gym`](../langgraph_orchestrator_reasoning_gym/) for new commands and configuration.

The `prepare.py` wrapper forwards to the renamed environment. NeMo Gym also resolves the former `config.yaml` path to the canonical configuration.
