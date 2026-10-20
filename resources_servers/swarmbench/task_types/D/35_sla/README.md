# 35 sla: SLA breach and service-credit audit (family D, procedural)

## Task
A fictional cloud provider's service-assurance team audits `n` customer contracts for the reporting month 2026-03.
For every customer the agent decides whether the availability commitment and the support first-response commitment
were breached, computes the measured monthly availability (three decimals) and the service credit owed.

**Unit = one customer contract for March 2026** (`CUS0001` ...). Evidence per unit:
- the contract: a code header (customer, fixed UTC offset, reporting month with its 44640 minutes) and 12 numbered
  clauses in shuffled order, each one of three K3-worded variants, with the parameters: target (99.95 / 99.90 /
  99.50 / 99.00), degraded weight (1, 0.5, 0), minimum incident length (0/5/10/15 min),
  maintenance notice hours (24/48/72) and weekly UTC window, upstream-provider failures counted or excluded,
  force-majeure and own-action exclusions, P1 calendar minutes and P2/P3 business-hour targets, support credit,
  three credit tiers, cap and claim deadline; plus the customer's fixed UTC offset;
- subscriptions (2-3 instances, each a service + region);
- the shared incident log (region-wide incidents that depend on the seed only, plus each customer's instance-level
  incidents and decoy incidents on other tenants' instances of the same service and region);
- maintenance notices (sent time, planned start/end), the customer's tickets and its credit claims.

Gold is a minute-exact simulation (`type.downtime`, interval sweep). The oracle (`solutions.py`) re-derives every unit
from the tools only, parsing the K3 clauses back with `common.tre`, with an independent minute-array implementation.

| trap | what the evidence looks like | careless reader |
|---|---|---|
| region-wide | incident scope `region-wide` on a subscribed service + region | searches by instance id only and misses it |
| decoy | instance-level incident on another tenant's instance, same service + region | counts it |
| maint_ok | notice early enough and planned period in the window; maybe overrun past planned end | counts it, or excludes the overrun |
| maint_late / maint_outside | notice too late, or planned on another weekday / past the window end | excludes it |
| degraded / overlap | degraded weight 1, 0.5 or 0; overlapping incidents | double counts, ignores weight |
| short | incidents shorter than the minimum (and one exactly at the minimum) | counts / drops wrongly |
| straddle | incident crosses 03-01 00:00 or 04-01 00:00 | counts minutes outside the month |
| own_action / force_majeure / upstream | cause-based exclusions; other tenants' customer_action counts | excludes all or none |
| cap | a major outage puts the tier credit (+ support credit) above a 20% cap | uncapped credit |
| tickets | business hours in the customer's offset, P1 on a weekend (calendar), ticket opened in February, UTC-vs-local business hours | calendar time, UTC hours, counts February |
| claims | late claim, claim for 2026-02 only, no claim | credit paid anyway |

Every unit has 2 planted availability traps, the first always an exclusion that changes its availability
(maint_ok, own_action, force_majeure, short when a minimum applies, upstream when excluded), plus region-wide exposure
(14 region-wide incidents and 6 region-wide maintenances per seed), half the units a plain counted outage, 20% a
cap-binding major outage, one ticket trap (40%: plus a real ticket breach) and a claim state. Availability is never within 0.003 of a tier threshold, tickets are never within 10-15 minutes
of their target, notice leads never tie the required hours, so each unit has one exact answer.

## Why one context cannot do the hard tier
A careful unit is ~11-13 tool calls (measured): contract, subscriptions, 2-3 incident searches by service + region (or one per
instance plus the region-wide check), 1-3 notices, tickets, claims and 1-3 calculator calls, then a minute-level union
of 4-10 intervals and business-hour arithmetic. Units split cleanly by customer (all world data is read-only).
See "Check 5" for the measured K3 time per unit; the hard tier is sized by volume from it.

## Tiers (n customers / budget s)
easy 6 / 1800, medium 16 / 1200, hard 60 / 1200. The final K3 single-unit loop took 143 s per unit on average
(median 132 s, 10.8 tool calls). Assuming OpenCode solo is 2-3x faster per unit (~57 s), hard is 60 x 57 s = 3400 s,
~2.9x the budget, and medium (16 x 57 s = 910 s) fits. Generation < 2 s at n = 60; a world is a prefix of a larger one
for the same seed (unit i depends only on (seed, i); region-wide events depend on the seed only).

## Tools
`list_customers(offset, limit)`, `read_contract(customer_id)`, `get_subscriptions(customer_id)`,
`search_incidents(service, region, instance_id, start_date, end_date, offset, limit<=40)` (rows: id, start and end
with weekday, logged minutes, service, region, scope, impact, cause, caused by, notice) (instance_id matches only
incidents that list it; region-wide rows are found by service + region), `read_notice(notice_id)`,
`list_tickets(customer_id)`, `list_claims(customer_id)`, `calculator(expression)`. Read-only, bounded to 8k chars,
cached per world under a lock; no tool computes availability, business hours or credits.

## Answer and scoring
One JSON object keyed by customer id: `{"CUS0001": {"breaches": ["availability", "support"], "availability": 99.871,
"credit": 10}}`. A unit is correct only if the breach set is exact, availability is within 0.0015 of gold and the credit
is exact (tolerance 0.0015 accepts truncation instead of rounding; one minute of downtime is 0.0022). `score` = fraction of customers correct; `score_partial` = mean of the three fields. Tolerant parsing:
code fences, case, "none", `"99.871%"`, `"10%"`.

## Checks (offline, `checks.json`, K3 text, n = 6 / 16 / 60, seeds 0, 1)
- 1 oracle (tool-only, parses K3 clauses, independent minute-array recomputation): 1.000 on all 6 instances.
- 2 plausible-wrong (max over instances): **regex 0.00** (grep the customer's instance ids in the log, full lengths,
  largest percentage as target, > 4 calendar hours = late, credit 10 on any breach), **naive reader 0.00** (right scope
  and clipping, but no exclusions, calendar support time, no claim or cap check). Service-region calendar reader
  (counts other tenants' instances, weight 1, calendar support, all maintenance excluded) 0.00-0.17 (0.00 at n = 60).
  Partial readers that skip one trap kind (reported, not required <= 0.1): region-wide 0.81, all maintenance excluded
  0.63, no maintenance excluded 0.73, degraded weight 0.94, minimum length 0.69, clipping 0.93, decoys 0.67, cause
  exclusions 0.67, business hours 0.68, claims 0.83; exclude-every-customer_action, UTC business hours, February
  tickets and cap reach 1.00 in some instance because those traps are rare per instance.
- 3 no-op: 0 for all five empty/format-only answers. 7: 0 tool errors on bad arguments; no tool computes downtime,
  business hours or credit. 8: deterministic hashes (n=60 seed 0: 0181bdcd4e57254e..., seed 1: 8f592818ba480881...).

## Check 5 (single units, self-hosted K3, `unitcheck.py`, 20 units, max 40 turns)
| run | world version | seeds, n | correct | mean s / unit | tool calls / unit |
|---|---|---|---|---|---|
| `_units/35_sla-k3` | v1: 3-4 traps, dates without weekday | 0,1, 60 | 11/20 | 210 | 12.8 |
| `_units/35_sla-k3v2` | v2: weekdays in timestamps, month minutes in header, 2-3 traps | 0,1, 40 | 9/20 | 135 | 13.3 |
| `_units/35_sla-k3v3` | v3: logged-minutes column, 2 traps (one exclusion), sparser region-wide events | 0,1, 40 | 15/20 | 194 | 11.3 |
| `_units/35_sla-k3v4` | **final**: thresholds printed with two decimals (99.50) | **2,3** (fresh), 40 | **18/20** | 143 | 10.8 |

Verifier audit: every failed unit of all four runs (37 of 80) was re-derived by hand with `_build/explain.py` (per
incident decision, notice lead, window check, business minutes, claim date). All 37 are model errors; 0 gold or
verifier errors (FP 0, FN 0 over 80 audited units). Error kinds: decimal threshold comparisons ("99.767 is below
99.5"; 3 of the 5 v3 failures, gone after the two-decimal print), weekday of a date (v1, fixed by printing weekdays),
missed region-wide incidents, excluding other tenants' customer_action incidents, duration/availability arithmetic,
claim deadline day counting, misread response dates. The final two failures: a missed region-wide maintenance with a
short notice plus a claim filed on day 16 counted as day 15; and an excluded other-tenant customer_action incident.
The changes between runs only removed reading friction (weekday, duration, number format, trap count); gold logic
and rules never changed.

## Provenance
- All task-facing prose (intro, rules, the three variants of each of the 16 contract clauses, unit prompt, tool
  descriptions) is K3-written: `text/k3.json`, produced by `k3gen.py` from `text/request.json` on the self-hosted
  Kimi-K3 pool (`policy_model`, moonshotai/Kimi-K3, self-hosted, 2026-10-14, one attempt, 234 s).
  The full prompt is `text/request.json["user"]`; the raw reply and usage are in `text/k3_raw.json`.
- `text/dev_stub.json` was written by Claude for offline tests only and never ships (`text_source` = k3).
- Claude wrote code only: ids, names, timestamps, parameters, tables (customer header line, log rows) are code.
- No external data. Canary line first in every prompt and in `world/CANARY.txt`.
