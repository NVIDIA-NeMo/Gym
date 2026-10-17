# VeriSoftBench agent

One rollout is one VeriSoftBench proof candidate and its repair chain: generate,
verify with `resources_servers/verisoftbench`, and on failure send upstream's repair
prompt as a fresh single-turn request (same system turn) up to `max_repair_rounds`
times, forwarding the previous round's proof, lemmas and error so the server can
apply upstream's fallbacks. `reward` is 1.0 if any round verified;
`solved_without_repair` and `repair_rounds_used` are added to the final response,
and `attempts` records every round.

## Configuration

- `max_repair_rounds`: repair rounds per failed candidate (upstream: 3). Must equal
  the resources server's `max_repair_rounds`.

See `resources_servers/verisoftbench/README.md` for the benchmark, setup and licensing.

## Tests

```bash
ng_test +entrypoint=responses_api_agents/verisoftbench_agent
```
