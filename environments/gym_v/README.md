# gym_v environment

Catch-all wiring for the Gym-V env-id-parametric resources server. Rows
in `data/example.jsonl` span a handful of env families (Games, Algorithmic,
Spatial, Graphs, Geometry). Adding more envs is data-only — append rows
to a curated JSONL, no server code change.

- **Resources server:** [`resources_servers/gym_v/`](../../resources_servers/gym_v/)
- **Agent:** [`responses_api_agents/gymv_agent/`](../../responses_api_agents/gymv_agent/)
- **Example dataset:** `data/example.jsonl` (8 rows).
