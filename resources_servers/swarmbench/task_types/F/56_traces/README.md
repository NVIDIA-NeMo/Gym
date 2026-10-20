# 56 traces: distributed trace analysis (family F)

## Task
`n` request classes (endpoint x client x region) of an online shop whose services and operations follow the
OpenTelemetry demo (open-telemetry/opentelemetry-demo, Apache-2.0: frontend, cart, checkout, currency, payment,
shipping, quote, email, product-catalog, recommendation, ad, accounting, fraud-detection, valkey-cart, flagd).
Each class has 8 traces. Per class, name the operation (`<service> <operation>`) with the largest mean self time
under exact rules: self time = duration minus the union of direct children's intervals clipped to the span; traces
whose root is ERROR are excluded; an operation's per-trace self time is summed over its spans; the mean is over the
class's remaining traces. Unit = one class. Score = fraction of classes right.

## Simulation (procedural, frozen per seed)
Eight user flows modelled on the demo's call graph, with parallel fan-out at the frontend, sequential calls in
checkout, and asynchronous Kafka consumers (accounting, fraud-detection) that start after their producer span ends.
Every call is a client span on the caller wrapping the callee's server span, as in real OTel traces, so the longest
span is always a wrapper and never the answer. One operation per class is planted 3-5x slower; 1-2 traces per class
are ERROR traces with a 2-4 s timeout on another operation, chosen so that including them changes the answer. A
class is redrawn until the winner beats the runner-up by at least 1.4x. Timings are computed once from the seed and
stored in the world; gold is recomputed from the stored spans.

## Tools
`list_classes(offset, limit<=100)`, `list_traces(class_id)`, `get_trace(trace_id)` (span rows).

## Tiers (n classes / budget s)
easy 24 / 1800, medium 384 / 1200, hard 1024 / 600 (about 9.5k chars of spans per class). The first settings (medium 96, hard 384 / 1200) gave Sol solo 1.00 and 0.99 and were replaced.

## Known limits
The union-versus-sum rule rarely changes the answer: "duration minus the sum of children" agrees with gold in
90-100% of classes, so it is kept as a diagnostic (`solutions.minus_sum`), not as a wrong solution. Knowing the
per-endpoint answer distribution would give about 0.41 without reading, but the agent has no access to it.

## Files
`type.py`, `solutions.py`, `tests/test_type.py`, `checks.json`.

## Provenance
All task-facing prose (prompt, rules, tool descriptions, phrase banks, unit prompts) was written by Kimi K3 (`nvidia/moonshotai/kimi-k3` on the NVIDIA inference API, served as `kimi-k3-high`) through `../k3gen.py`, with the system prompt below, and frozen in `text/`. Claude wrote only code, schemas, verifiers and tests. Code fills ids and numbers into K3's `{placeholders}`.

### `text/k3.json` (sha256 ad9c1728d9ede4ee..., created 2026-10-14 06:51:24 UTC, served model nvidia/moonshotai/kimi-k3, 1 attempt(s))

System prompt:

```
You write task text for an offline evaluation of software agents. Follow the facts you are given exactly; do not add rules, numbers or names that are not in the brief. Reply with exactly one JSON object and nothing else.
```

User prompt:

```
Write the task-facing text for a distributed-trace analysis task. The agent sees traces of an online-shop microservice system in the style of the OpenTelemetry demo (services such as frontend, cart, checkout, currency, payment, shipping, quote, email, product-catalog, recommendation, ad, accounting, fraud-detection, valkey-cart, flagd). Requests are grouped into {n} request classes (endpoint, client and region), each with {k} sampled traces. Every span row has span_id, parent_id (- for the root span), service, operation, start_us, duration_us and status. Return one JSON object with these keys:

"intro": the opening paragraph, with the placeholders {n} and {k}. The job: for every request class, find the operation (identified by service and operation name together) with the largest mean self time under the rules below.
"rules": a list of bullet strings stating these exact rules, one per bullet, in this order, keeping every condition exact:
 1. The self time of a span is its duration minus the part of its own time interval [start_us, start_us + duration_us) covered by its direct children; children that run in parallel are counted once (take the union of the children's intervals), and any part of a child that lies outside the parent's interval does not count (asynchronous consumers can start after their parent has ended).
 2. A trace whose root span has status ERROR is excluded from all averages; status values of other spans do not matter.
 3. An operation's self time in a trace is the total self time of all its spans in that trace (an operation can appear several times in one trace).
 4. An operation's mean self time in a class is the average, over the class's non-excluded traces, of its self time per trace.
 5. The answer for a class is the operation with the largest mean self time. Operations are identified as "<service> <operation>", for example "cart oteldemo.CartService/GetCart". The slowest span by total duration is usually a parent and is often not the answer.
"tools": an object with one-sentence descriptions for list_classes (class ids with their endpoint, client and region, paged, at most 100 per call), list_traces (the trace ids of one class with each root span's status and duration and the span count), get_trace (all spans of one trace as rows: span_id | parent_id | service | operation | start_us | duration_us | status).
"answer": a short paragraph: the final answer is one JSON object mapping every class id to its operation written as "<service> <operation>", for example {example}.
"score": one sentence: the score is the fraction of classes answered correctly.
"unit_prompt": instructions for a single worker who must analyse exactly one class, {cid}, with the same tools and the rules placed at {rules}; the worker must end with exactly one final line of the literal form UNIT | {cid2} | <service> <operation>. ({cid2} is the same id as {cid}.)
```

