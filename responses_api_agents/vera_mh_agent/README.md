# vera_mh_agent

Conversation simulator for VERA-MH: a persona model server plays the user, the policy model server plays the
chatbot, and the resources server `vera_mh` judges the transcript. See
[`resources_servers/vera_mh/README.md`](../../resources_servers/vera_mh/README.md); the instance config lives in
`resources_servers/vera_mh/configs/vera_mh.yaml`.

Config keys: `model_server` (the chatbot under evaluation), `user_model_servers` (map from the dataset row's
`user_simulator` to a model server), `user_responses_create_params` (optional per-simulator overrides), `max_turns`
(30; rounded so the chatbot speaks last), `persona_speaks_first`, `start_prompt`, `termination_signal`.

The loop, message construction and transcript format port upstream's `generate_conversations/conversation_simulator.py`
and `utils/conversation_utils.py` (commit `2c9d1fc`). A simulator or chatbot call that fails returns a failures-sidecar
row (`vera_mh_simulation_failed`) rather than a scored conversation.

```bash
pytest responses_api_agents/vera_mh_agent/tests/
```
