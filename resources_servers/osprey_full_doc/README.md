# Osprey `full_doc` resources server

Verifier for the Osprey Document AI extraction benchmark inside NeMo Gym.

The original Osprey evaluator is deterministic:

- `API error`
- `Extraction error`
- `False positive`
- `False negative`
- `Incorrect value`

This Gym resource server preserves those same categories while using the
standard `tool_simulation_agent` benchmark path, so tools are not executed
during rollout.
