# ASB Modal Campaign — status

**Superseded.** The adapter described here is built. See:

- [`benchmarks/asb/README.md`](benchmarks/asb/README.md) — layout and usage
- [`benchmarks/asb/METRICS.md`](benchmarks/asb/METRICS.md) — denominators, metrics, deviations

## Correction to the earlier handoff

The previous version of this file stated that the complete expansion is **2,040 selectors
per attack condition** and warned against the runner's `task_num=1` default. **Both are
wrong, and following them would have produced numbers comparable to no published ASB
result** — at five times the cost.

`main_attacker.py` defaults `--task_num` to 1, and `scripts/agent_attack.py` — the driver
behind the DPI, OPI, MP and mixed tables — never overrides it. Each published condition is
therefore `10 agents x 1 task x 40 agent-matched attacker tools = 400 rows`.
`config/POT.yml` is the only config that sets it explicitly (`task_num: 2`), over a
5-agent task file: `5 x 2 x 40 = 400` as well.

Confirmed three independent ways:

1. Every entry in both published defense tables is a multiple of `0.25% = 1/400`. The main
   table's DPI/OPI/MP columns are multiples of `0.05% = 1/2000` (five attack types x 400).
   Mixed resolves only on a denominator divisible by three — Gemma2-9B's 92.17% is
   `1106/1200` — matching the three uncommented attack types in `config/DPI.yml`.
2. Upstream's shipped Chroma memory stores hold ~400 documents each, one per row of the
   DPI run that wrote them.
3. The expansion lands on exactly 400 for all 27 conditions, PoT included.

The 2,040 figure is the full 51-task cross product, which no published run used. It
remains reachable via `--task-num` and is not the public benchmark.

## Still true from the earlier handoff

- Upstream pinned at `1f561dccf92d55302368fa67679b4ba9d9c8fdc4`; never vendored.
- `data/agent_task_pot_all.jsonl` is GitHub rate-limit HTML rather than JSONL. No config
  references it. `prepare.py` asserts it is still HTML so an upstream fix surfaces as a
  test failure instead of silently changing the benchmark.
- All four target endpoints authenticate as documented; Super VL uses a separate token.
- Provider and infrastructure failures stay in sidecars, outside quality denominators.

## Not done

- **Hugging Face dataset push.** `python -m benchmarks.asb.prepare push` is implemented and
  tested but needs an `HF_TOKEN`; none is present on this machine. Until it runs, the
  pinned rows exist only locally under `resources_servers/asb/data/` (gitignored), with
  their content hash recorded in the committed `manifest.json`.
