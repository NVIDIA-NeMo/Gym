# 30 inventory: inventory roll-forward across warehouses (family C, procedural)

## Task
A fictional distributor runs `n` warehouses (`WH001`...). For the period 2026-01-01 to 2026-03-31, the agent must
report the ending on-hand quantity in eaches of every SKU each warehouse stocks. Inputs per warehouse: opening
balances (physical count at the end of 2025-12-31), a transaction journal (extract taken 2026-04-10, so it holds
lines posted after the period), and a page of warehouse notes. Shared inputs: the item master (case pack sizes with
the dates they took effect) and the inventory accounting policy (K3 text, every rule stated exactly).

**Unit** = one warehouse (4-8 SKUs, ~31 journal lines, ~2.9k characters of journal + notes). A unit is correct only
if every SKU of the warehouse is exact. Gold is computed by simulation (`type.roll`); the tool-only oracle rebuilds
it from the rendered text.

## Traps (policy rules a careless reader gets wrong)
Every warehouse has a backdated line (posted after the cutoff, effective before it) plus 4-6 more of these kinds,
always including at least one note correction:
- **case conversion with pack-size changes**: CS lines convert at the pack size in effect on the effective date
  (about half the SKUs change pack size mid-period);
- **effective-date cutoff**: future-dated lines posted before the cutoff do not count; backdated lines do;
- **cycle count**: a COUNT resets the SKU as of its effective date; a receipt effective before the count but posted
  after it is already in the count;
- **voids**: a VOID cancels the referenced line;
- **returns by condition**: restocked only if the memo (K3 phrase) says the goods are resaleable as new, else scrap;
- **transfers in transit**: the sender removes an XFER-OUT only if the matching XFER-IN (in the *receiving*
  warehouse's journal, found with `search_journal`) is effective on or before the cutoff; the receiver adds what it
  received (short receipts happen); some transfers are never received in the extract;
- **notes** (K3 templates): receipt booked to the wrong SKU, consignment stock that does not count, duplicate
  posting to ignore.

Transfers make warehouses interdependent within blocks of four (`BLOCK`), which keeps generation prefix-stable:
the first `4k` warehouses of a larger instance are identical to a `4k`-warehouse instance with the same seed.

## Why one context cannot do the hard tier
A careful unit is ~10-14 tool calls (journal page, notes, item rows for pack changes, 1-2 transfer searches,
calculator calls) and ~4-6k tokens of tool output. Measured cost is lower than first estimated: single-unit K3 runs
took 155 s and 12 tool calls per warehouse, but an OpenCode solo K3 probe at n=40 / 1200 s scored 0.975 (39/40;
26 model calls, 86 tool calls), about 32 s per warehouse, because one agent batches reads and arithmetic across
warehouses. The hard tier is therefore sized at 120 warehouses (~700 SKU balances, ~3,700 journal lines): at
~32 s per warehouse that is ~3,800 s, about 3x a 1200 s budget, while the warehouses split cleanly between workers
(only transfer receipts are looked up across warehouses).

## Tiers (n warehouses / budget s)
easy 8 / 1800, medium 40 / 1200, hard 120 / 1200 (multiples of the 4-warehouse transfer blocks). The first sizing
(5 / 12 / 40) was too easy for solo K3 (0.975 at 40 / 1200 s). Rule used: n x (solo s per unit) ~ 3x budget at
hard; medium 40 is the probed point where solo still finishes.

At n=120: generate 0.03 s; list_warehouses 3.5k chars (one page), list_items 40 SKUs per page (max 3.3k chars, 334
SKUs over 9 pages), read_journal max 4.2k chars (max 43 lines, one 50-line page), read_notes max 0.7k, policy 2.2k;
each transfer id returns at most 2 search hits, so the 20-hit cap does not bite.

## Tools (read-only, offline, outputs <= 8k chars)
`list_warehouses(offset, limit<=200)`, `read_policy()`, `list_items(offset, limit<=40)`, `read_item(sku)`,
`read_journal(warehouse, offset, limit<=50)` (first page starts with the heading and opening balances),
`search_journal(text)` (substring over all journals, 20 hits), `read_notes(warehouse)`, `calculator(expression)`.

## Answer and scoring
One JSON object: `{"WH001": {"SKU-1000": 1830, "SKU-1003": 412}, ...}` (last JSON object with `WH` keys is graded;
numbers may be strings). If no complete answer object parses (for example a truncated answer), every complete
`"WHnnn": {...}` entry is graded on its own (a later entry for the same warehouse replaces an earlier one);
malformed or missing warehouses count as wrong, and empty or format-only answers still score 0. `score` = fraction of warehouses with every SKU exact and no extra SKUs;
`score_partial` = fraction of SKU cells exact; `score_exact` = all warehouses right.

## Baselines (check 2, current `checks.json`, K3 text, n = 8/40/120, seeds 0-1, max over instances)
| strategy | score |
|---|---|
| naive signed sum of raw quantities | 0.00 |
| spreadsheet: posting-date cutoff, no notes, no counts, transfers at ship | 0.00 |
| careful but posting-date cutoff and no notes | 0.00 |
| partial reader, posting-date cutoff | 0.00 |
| partial reader, no case conversion | 0.00 |
| partial reader, ignores notes | 0.025 |
| partial reader, first pack size only | 0.125 |
| partial reader, ignores cycle count | 0.45 |
| partial reader, transfers removed at ship | 0.617 |
| partial reader, all returns restocked | 0.75 |
| partial reader, ignores voids | 0.75 |

Oracle 1.0, empty/format-only answers 0.0, 0 tool errors on bad arguments, deterministic hashes, text_source k3.

## Verifier audit (check 4, single units)
20 single-unit runs with self-hosted K3 (n=12, seeds 0-1; `../_units/30_inventory-k3/units.jsonl`): 19/20 correct,
mean 155 s and 12 tool calls per unit. All 20 verdicts were checked by hand against the world (`audit/units_audit.json`):
the 19 correct answers match the policy result on every SKU (5-8 per unit), and the one failure (WH003, seed 1) is a
true negative: a 13 CS receipt effective 2026-02-03 was converted at the old pack size 12 instead of 10 (effective
from 2026-01-27), so one SKU is off by 26. **FP 0, FN 0.** No ambiguity was found and nothing was changed.

Calibration rollouts (`../_results/rows-calA.jsonl`, 18 runs, seed 9100; all answers regrade identically):
10 graded-wrong warehouses from medium/hard runs and 5 graded-right ones were checked by hand against the world
(details in `audit/units_audit.json`). Every wrong one is a true model error: a dropped journal line (the same
WH014 receipt is missed in three runs), a case converted at the wrong pack size (including on the change date),
a duplicate note ignored, and one arithmetic slip that gave a negative balance. Every right one was recomputed by hand
and holds. **FP 0, FN 0.** Two solo runs (n=40 and n=120) scored 0 with no final answer (timeout, narration
only); three other solo runs ran out of budget after listing 26/40, 40/120 and 55/120 warehouses. Cosmetic only:
voided and duplicated SHIPMENT lines carry a `PO-` reference (no effect on gold); nothing was changed.

## Files
`type.py` (generate/tools/score, `roll` = the policy), `solutions.py` (tool-only oracle, baselines, single units),
`tests/test_type.py`, `checks.json`, `text/request.json` (K3 brief), `text/dev_stub.json` (Claude-written stub for
local tests only; never ships).

## Sources
Procedural (P). No external data. Product descriptions, cities and vendor names are code word lists.

## Provenance
All task-facing prose (prompt, policy, headings, notes, return-reason memos, tool descriptions, unit prompt) comes
from `text/k3.json`, to be written by self-hosted Kimi K3 with `../k3gen.py` from `text/request.json`.
`text/k3.json` sha256 5ec5fd3c04905e01..., created 2026-10-14 13:04:58 UTC by self-hosted Kimi K3 (vLLM, served as
`policy_model`, endpoint http://10.49.69.30:8000/v1), 1 attempt; full prompt and raw reply in `text/k3_raw.json`.
