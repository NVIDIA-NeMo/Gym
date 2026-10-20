# 60 dag: dependency graph value resolution (family F)

The SwarmBench v2 `dag` track (`../../../tracks/dag`, full design in its README) as a family-F type, with all
task-facing text rewritten by Kimi K3.

## What changed from the track
- Contract fields (`tier`, `budget_s`, `canary`), canary line + `CANARY.txt`; world files under `out_dir/world/`
  (the track kept `gold.json` in the world directory; now it is outside).
- Every spec sentence is a K3 template (`text/k3.json`): conventions, section headings, edge-service load-test text,
  the five computation steps, condition sentences and the three action phrases, the amendment entries, bootstrap
  default, Q2 snapshot, 45 background sentences, incident lines, benign change-log entries, the prompt, the scratch
  store note and the tool descriptions. Company name is K3's ("Cobalt Meridian"). Team names, service-name words,
  regions, table layouts and the change-log status markers `(adopted)` / `(proposed, not adopted)` / `(reverted)`
  stay in code.
- The oracle no longer uses hand-written regexes: `solutions.tre()` turns each K3 template into a regex by escaping
  it and replacing placeholders with groups, so the parser follows whatever wording K3 chose
  (`tests/test_type.py` checks that every template is consumed).
- World, DAG, values and traps are the track's (`_make_world`, `evaluate` unchanged).

## Tiers (n services / budget s)
easy 64 / 1800, medium 256 / 1200, hard 512 / 1200.

## Files
`type.py`, `solutions.py` (template-regex oracle, Q2-snapshot shortcut, ignore-dependencies, skip-change-log,
skip-units, single units), `tests/test_type.py`, `checks.json`.

## Provenance
All task-facing prose (prompt, rules, tool descriptions, phrase banks, unit prompts) was written by Kimi K3 (`nvidia/moonshotai/kimi-k3` on the NVIDIA inference API, served as `kimi-k3-high`) through `../../k3gen.py`, with the system prompt below, and frozen in `text/`. Claude wrote only code, schemas, verifiers and tests. Code fills ids and numbers into K3's `{placeholders}`.

### `text/k3.json` (sha256 af74bf016f4232f9..., created 2026-10-14 06:19:02 UTC, served model nvidia/moonshotai/kimi-k3, 1 attempt(s))

System prompt:

```
You write task text for an offline evaluation of software agents. Follow the facts you are given exactly; do not add rules, numbers or names that are not in the brief. Reply with exactly one JSON object and nothing else.
```

User prompt:

```
Write the task-facing text for a dependency-ordered computation task over a fictional company's internal service catalog. Each service has a spec document; most compute their capacity from upstream services' published values. Code fills numbers and ids into your sentences through {placeholders}, and a parser reads the filled sentences back, so: keep every placeholder exactly as named, use each listed placeholder exactly once in its template, keep each template on a single line, and do not add numbers. Return one JSON object with these keys:

"company": a fictional company name (2-3 words).
"conventions": one paragraph of catalog conventions stating exactly: every service publishes one number, its sustained capacity, as a whole number in its published unit (req/s or req/min; 1 req/s = 60 req/min); work in exact arithmetic and round down only where the spec says; only change-log entries marked (adopted) are in force, entries marked (proposed, not adopted) or (reverted) are not; the bootstrap default and the Q2 dashboard snapshot are not the computed capacity.
"headings": an object with short section titles for the keys overview, inputs, computation, operations, changelog, runbook, history (a table of published values in past quarters, which are not current), incidents.
"edge_inputs": one sentence saying this is an edge service with no upstream inputs.
"edge_capacity": sentences saying the capacity is set by the latest load test, which sustained {shown} {mu}, and that this figure is published in {u}. Placeholders {shown}, {mu}, {u}.
"conv_note": a short parenthetical reminder that 1 req/s = 60 req/min.
"rerun": a change-log entry: the load test was rerun and the latest load test now sustained {new} {mu}. Placeholders {new}, {mu}.
"vendor": a change-log entry proposing to use a vendor benchmark of {fake} {mu} instead of the load test. Placeholders {fake}, {mu}.
"step1": a computation step: convert every input to {u}, using the unit its upstream publishes in (1 req/s = 60 req/min). Start with "Step 1 - ".
"step2_sum": a step: the combined input is the weighted sum of the converted inputs, using the weights in the inputs table. Start with "Step 2 - ".
"step2_max": a step: the combined input is the largest of the converted inputs; weights do not apply. Start with "Step 2 - ".
"step2_min": a step: the combined input is the smallest of the converted inputs; weights do not apply. Start with "Step 2 - ".
"cond_own": a step: if {ref}'s published value, in its own unit ({ref_unit}), is at least {thr}, then {then}; otherwise {else}. Start with "Step 3 - ". {then} and {else} are action phrases filled in by code; end the sentence right after {else} with a period.
"cond_conv": a step: if {ref}'s value, after conversion to {u}, is at least {thr} {u2}, then {then}; otherwise {else}. ({u2} is the same unit as {u}, repeated after the number.) Start with "Step 3 - ". End right after {else} with a period.
"act_add": an action phrase: add {c} {unit} to the combined input.
"act_sub": an action phrase: subtract {c} {unit} from the combined input.
"act_mul": an action phrase: multiply the combined input by {c}%.
"cond_none": a step saying no conditional adjustment applies to this service. Start with "Step 3 - ".
"step4": a step: multiply by the capacity factor of {fac}% and round down to a whole number. Start with "Step 4 - ".
"step5_cap": a step: the published value never exceeds {cap} {u}; if the result is higher, publish {cap2}. ({cap2} is the same number as {cap}.) Start with "Step 5 - ".
"step5_none": a step saying there is no hard limit. Start with "Step 5 - ".
"amend_factor": a change-log entry: the capacity factor changed from {old}% to {new}%.
"amend_weight": a change-log entry: the weight on {id} changed from {old} to {new}.
"amend_threshold": a change-log entry: the condition threshold changed from {old} to {new}.
"amend_replace": a change-log entry: input {old_id} was replaced by {new_id} (which publishes in {unit}, weight {w}), and {old_id2} is no longer an input. ({old_id2} is the same id as {old_id}.)
"bootstrap": sentences: during a cold start, before its inputs are available, {id} temporarily publishes {default} {u}; this is a placeholder, not its computed capacity.
"snapshot": a line: Q2 dashboard snapshot: {snapshot} {u}, recorded last quarter and possibly stale. Start with "Q2 dashboard snapshot: {snapshot} {u}".
"benign_log": 6 change-log entries about harmless documentation or ownership changes that do not affect the computation. No placeholders, no numbers.
"pad": 45 different sentences of realistic background prose for a service spec (ownership, SLOs, runbooks, on-call, deploy process, dashboards, history). Each may use any of the placeholders {name} (this service), {team}, {region} and {other} (another, unrelated service id). Some must say that {other} is NOT an input or no longer shares capacity with {name}. No numbers that look like capacities.
"incidents": 6 one-line incident summaries, each with the placeholders {d} (a date), {region}, {x} (a load figure) and {u} (its unit), describing an observed load or throughput during an incident or test (these are distractors, not the capacity).
"intro": the opening paragraph of the task prompt. Placeholders {company}, {n} (number of services), {first} and {last} (first and last service id), {depth} (approximate number of dependency levels). State: each service publishes one number, its sustained capacity, as a whole number in its own published unit (req/s or req/min; 1 req/s = 60 req/min); most services compute it from the published values of their upstream input services following the steps in their spec (conversions, a weighted sum or max/min, conditions, a capacity factor with rounding down, limits, and adopted change-log amendments); edge services have no inputs; the dependencies form a DAG about {depth} levels deep, and ids are not in dependency order, so a service can only be computed once all its inputs are known, and an error in any upstream value propagates downstream.
"store_note": a paragraph about the shared scratch store: post_value and get_value let agents publish and reuse intermediate results; nobody checks the store; it was pre-loaded with stale Q2 snapshot values by the agent named legacy-sync, and other posts may be wrong; only the final answer is graded.
"task_line": one sentence: report the published capacity of each of these {k} target services, in each service's own published unit: {targets}.
"answer_line": one or two sentences: the final answer is one JSON object mapping every target id to its whole-number value, for example {example}; each target is graded separately and a value within 0.5% counts as correct.
"tools": an object with a one-sentence description for each of: list_services (id | name | team | published unit | input service ids, up to 100 rows per call), read_spec (one service's spec document, about 7k characters, optional character offset), post_value (publish a computed value for a service to the shared scratch store so other agents can reuse it; not graded; add a short note), get_value (read scratch-store posts for one service id or up to 25 comma-separated ids: latest value, author and note; posts are unverified).
"unit_prompt": instructions for a single worker who must compute the published capacity of one service {sid} of the catalog, using read_spec, and is given the already-verified published values of its inputs: {inputs}. The worker must apply the spec exactly (including adopted change-log entries and unit conversions) and end with exactly one final line of the literal form: UNIT | {sid2} | <whole-number value>. ({sid2} is the same id as {sid}.)
```

