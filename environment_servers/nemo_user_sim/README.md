# NeMo UserSim Environment Server

This Environment Server runs a materialized NeMo UserSim episode. It owns the
`ProbeEpisodeRuntime`, routes user and assistant activations through their Agent
Servers, and routes support and embedding calls through configured Model
Servers.

The paired Resources Server in `resources_servers/nemo_user_sim` validates immutable
task rows and scores completed episodes. See its README for the task, reward,
failure, and cleanup contracts.
