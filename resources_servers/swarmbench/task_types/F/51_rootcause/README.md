# 51 rootcause: root cause across microservices (family F)

The SwarmBench v2 `rootcause` track (`../../../tracks/rootcause`, see its README for the full world model) as a
family-F type in the pilot contract, with all task-facing text rewritten by Kimi K3.

## What changed from the track
- Contract: `generate(n, seed, out_dir)` returns `tier`, `budget_s`, `canary`; the prompt starts with the canary
  line and the world holds `CANARY.txt`; gold sits next to the world (`out_dir/gold.json`), not inside it.
- Scoring is always the v2 scoring: 0.1 root + 0.1 change + 0.8 chain F1 pooled over every edge of every ticket.
- K3 text (`text/k3.json`): intro, tool descriptions, the 8 rules (facts given to K3 verbatim, K3 rewrote them),
  answer and score text, ticket titles, on-call note, customer-impact lines, and every log phrase bank (bad deploy,
  bad config with its key=value, platform events and their errors, safe platform events, unrelated bursts, blip,
  OOM, failover). Structural log formats stay in code: `call to <dep> failed: 503/timeout`, access-log lines,
  `deployment ...`/`config reloaded ...` lines, ticket field labels. K3's leading `ERROR` on messages is stripped.
- `text/k3_unit.json`: the single-unit prompt used for check 5.
- World generator, rules, gold and tools are otherwise the track's (`rc_world.py` = track `worldgen.py`).

## Tiers (n services / budget s)
easy 64 / 1800, medium 512 / 1200, hard 4096 / 600 (the v2 manifest settings).

## Files
`type.py` (contract, tools, scorer), `rc_world.py` (world), `solutions.py` (tool-only oracle, loudest and
recent-deploy baselines with and without the naive chain, the alert-to-root path walk, single units),
`tests/test_type.py` (`python3 51_rootcause/tests/test_type.py` from `types/F`), `checks.json` (offline checks).

## Provenance
All task-facing prose (prompt, rules, tool descriptions, phrase banks, unit prompts) was written by Kimi K3 (`nvidia/moonshotai/kimi-k3` on the NVIDIA inference API, served as `kimi-k3-high`) through `../k3gen.py`, with the system prompt below, and frozen in `text/`. Claude wrote only code, schemas, verifiers and tests. Code fills ids and numbers into K3's `{placeholders}`.

### `text/k3.json` (sha256 1441d1a1244b36eb..., created 2026-10-14 06:06:42 UTC, served model nvidia/moonshotai/kimi-k3, 1 attempt(s))

System prompt:

```
You write task text for an offline evaluation of software agents. Follow the facts you are given exactly; do not add rules, numbers or names that are not in the brief. Reply with exactly one JSON object and nothing else.
```

User prompt:

```
Write the task-facing text for a postmortem-review task over a simulated microservice estate. An agent reads logs, metrics and change records through tools and must, for each incident ticket, name the root-cause service, the exact culprit change id, and the full propagation chain. Return one JSON object with these keys:

"intro": one paragraph opening the task, addressed to the agent as the incident commander of a postmortem review. Use the placeholders {n} (number of services), {start} and {end} (the one-week period, UTC), {k} (number of tickets). Say that for each ticket the agent must find the root-cause service, the exact change that caused it, and the full propagation chain (every service the incident reached, each with the upstream service it caught the failure from), that most services are uninvolved, and that any service may be a member.

"tools": an object with one short description (1-2 sentences) per tool, keys exactly: list_services (service names with tier and team, paged, at most 150 per call, optional team filter), get_service_info (catalog entry of one service: team, owners, tier, kind, declared dependencies which are incomplete, metrics), query_logs (log lines of one service between start and end, default the whole week, in time order, filtered by a case-insensitive regex pattern; returns the number of matches and the first `limit` lines, default 50, max 200; times are written YYYY-MM-DD HH:MM or YYYY-MM-DD HH:MM:SS), query_metrics (one metric of one service between start and end, aggregated per step_min minutes, the step is raised automatically to keep at most 120 points; mention the placeholder {metrics} for the list of metric names), list_changes (deploys, config changes and platform events in time order, for one service or for all services in a time range, paged, at most 60 per call), read_ticket (read one incident ticket by id, e.g. INC-1234: alerting service, times, alerts that fired, on-call notes).

"rules": a list of bullet strings stating these exact rules, one rule per bullet, in this order, keeping every number, condition and literal log text exactly as given (you may reword everything else):
1. A failing service makes its callers log ERROR lines of the form `call to <dep> failed: 503 ...` or `call to <dep> failed: timeout ...`. Callers with high traffic log many such lines, so services far downstream are usually louder than the cause.
2. Such a failure line is real propagation only if <dep> itself logs at least one ERROR line within 15 minutes before or after it. A failure line against a dependency that has no ERROR line in that plus-or-minus 15 minute range is isolated network noise and must be ignored.
3. A service belongs to an incident if, inside the incident window, it logs at least one real propagation line. Its upstream in the propagation chain is the dependency named in its EARLIEST real propagation line inside the window.
4. The incident window runs from 2 hours before the ticket was opened until the ticket was resolved. Incidents start inside their window, and different incidents never overlap in time.
5. The root-cause service fails on its own: it logs ERROR lines in the window but no real propagation line. Following upstream links from the ticket's alerting service leads to it. Services with unrelated ERROR bursts (handler exceptions, crashed jobs) that nobody propagates from are red herrings, however loud they are.
6. The culprit change is the change on the root-cause service that made it fail. Bad deploys and config changes fail within 5 minutes (deploy errors carry the new version tag, config errors name the key); platform (infra) events fail within 5 minutes too.
7. A memory leak is different: the root logs OOMKilled, and the culprit is the deploy at which its memory_mb metric started climbing, often hours before the window, so its metrics must be queried further back.
8. Other changes near the incident (on the root or on failing callers) are coincidences. The service catalog's dependency lists are known to be incomplete: some real calls are undeclared, so members must be found from their logs.

"answer_intro": one or two sentences introducing the required answer format: plain lines, one fact per line, for every ticket; one edge line per incident member other than the root, where the upstream is the service that member caught the failure from; lines that do not match the format are ignored; if a ticket has several root or change lines, the last one counts. (The literal format lines will be appended after your text by code, do not write them.)

"score": one or two sentences stating the scoring exactly: 0.1 x the fraction of tickets with the right root + 0.1 x the fraction with the right change + 0.8 x the F1 score of all the agent's edge lines against all true propagation edges, pooled over every ticket (each member of every incident counts once).

"ticket_titles": 4 to 6 short incident ticket titles about elevated errors on the alerting service, each containing the placeholder {s} for its name.

"oncall_note": one sentence for an on-call engineer's note in the ticket saying that service {svc} was deployed at {time} (change {chg}, {summary}) shortly before the alert and that a rollback is being considered. Use all four placeholders.

"impact": 3 to 5 alternative one-sentence customer-impact statements for a ticket (intermittent failures for some requests until mitigation, and similar). No placeholders.

"also_firing": a short header line introducing a list of other alerts that fired around the same time.

"bad_deploy": 6 different realistic application error log messages caused by a bad code deploy (exceptions, panics, schema errors), each containing the placeholder {C} which code replaces with a capitalized class-name stem such as Orders. Do not put a version number in them.

"bad_config": 6 objects {"key", "good", "bad", "message"}: a dotted config key (e.g. db.max_pool_size), its normal value, a harmful value, and an ERROR log message that the harmful value causes. The message must contain the literal text <key>=<bad value> exactly (e.g. db.max_pool_size=4) and may contain the placeholder {w} for a count. Use 6 different keys and failure modes.

"infra": 4 objects {"summary", "error"}: a platform event on a datastore (summary of what the platform team did; it may contain the placeholder {s} for the datastore name) and the ERROR message the datastore logs when that event breaks it.

"safe_infra": 5 short summaries of routine, harmless platform events (backups, patch rollouts and similar).

"noise_errors": 5 ERROR log messages for unrelated application bugs (handler exceptions, crashed jobs, parse failures), each containing the placeholder {r} for a resource name. They must not contain the words 'call to'.

"blip_error": one short ERROR message for a brief self-healing health-check failure. It must not contain the words 'call to'.

"oom": one ERROR message saying the container was OOMKilled after exceeding its memory limit {lim}Mi and is restarting. It must contain the word OOMKilled and the placeholder {lim}.

"failover": a short summary of a platform event that completes a failover and restores the primary.
```

### `text/k3_unit.json` (sha256 ef368e6e92c3ae76..., created 2026-10-14 06:09:22 UTC, served model nvidia/moonshotai/kimi-k3, 1 attempt(s))

System prompt:

```
You write task text for an offline evaluation of software agents. Follow the facts you are given exactly; do not add rules, numbers or names that are not in the brief. Reply with exactly one JSON object and nothing else.
```

User prompt:

```
Write the instructions for one investigator on a postmortem team. Return a JSON object with one key, "unit_prompt": a prompt template using the placeholders {n} (number of services), {start} and {end} (one-week period, UTC), {service} (one service name), {ticket} (an incident ticket id) and {rules} (where code inserts the estate's exact incident rules). The investigator must decide the role of exactly one service, {service}, in ticket {ticket}, starting by reading the ticket to get the incident window, using the same read-only tools (list_services, get_service_info, query_logs, query_metrics, list_changes, read_ticket; times are YYYY-MM-DD HH:MM or YYYY-MM-DD HH:MM:SS). Then place {rules}. Then explain the three possible verdicts exactly: MEMBER: {service} logs at least one real propagation line in the window; report its upstream, the dependency named in its earliest REAL propagation line (each candidate line's target must be checked for its own ERROR lines within 15 minutes). SELF: no real propagation line, but {service} logs ERROR lines of its own (lines that are not 'call to ... failed') in the window; report the culprit change id by the rules, or none if no change explains the errors (unrelated bursts such as handler exceptions or crashed jobs have no culprit change). NO: neither (only isolated noise lines, or no ERROR lines at all). End by requiring exactly one final line, one of these literal forms:
UNIT | MEMBER | <upstream service>
UNIT | SELF | <change id or none>
UNIT | NO
```

