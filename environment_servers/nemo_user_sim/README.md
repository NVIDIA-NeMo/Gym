# NeMo User Sim Environment Server

This Environment Server runs a materialized NeMo User Sim episode. It owns the
`ProbeEpisodeRuntime`, routes user and assistant activations through their Agent
Servers, and routes judge, summary, and tool-simulation calls through configured
Model Servers. Embedding calls go directly to `/embeddings` on the endpoint
configured for `embedding_model`, using its `base_url` and `api_key`.

Before execution, it asks the paired Resources Server in
`resources_servers/nemo_user_sim` to validate and snapshot the materialized row.
After execution, it sends the episode result back for scoring. The Resources
Server compares the row digest at verification time, establishing that the row
did not change during the episode.

Unexpected runtime errors become failures-sidecar records with
`_ng_failure_stage: "agent"`. Assistant-attributed failures that NeMo User Sim
handles are returned as measured episode results instead.

See the Resources Server README for the task, reward, failure, and cleanup
contracts.
