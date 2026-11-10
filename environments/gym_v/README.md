# gym_v environment

Catch-all wiring for the Gym-V env-id-parametric resources server. Rows
in `data/example.jsonl` span a handful of env families (Games, Algorithmic,
Spatial, and Graphs). Adding another environment reuses the same process pair,
but it must also supply an audited typed `success_criterion`; Gym-V rewards and
termination flags are not universally binary. The checked-in server config
rejects rows outside its current five-environment success allowlist.

- **Resources server:** [`resources_servers/gym_v/`](../../resources_servers/gym_v/)
- **Agent:** [`responses_api_agents/gymv_agent/`](../../responses_api_agents/gymv_agent/)
- **Example dataset:** `data/example.jsonl` (5 rows, matching the resources-server publication fixture).
