# Testing / Validation

## Checker integration history

Integration testing combined checker commit `2962b195f` with the changes from
[#5007](https://github.com/NVIDIA-NeMo/Gym/pull/5007) and passed all 526 `partner_if`
tests. PR #5007 has since merged into `main`. This branch now merges current `main`,
preserving its SWE integration and the new advertised-tool checker.

## Dataset pin

The registry artifact is pinned to `partner_if 2.1.5` (27 rows). Upload and download
were verified byte-for-byte against the recorded SHA-256. This branch routes the
new next-call row through the existing text environment and includes the three SWE
routes from `main`. Full-cohort coverage remains pending only the separately
maintained native Prime integration. The route-coverage test remains a readiness gate.

Before merging `main`, the component suite passed 495 tests and data validation;
preparation tests passed 5 tests with the full-release route-coverage test failing.
Those results are historical; the post-merge results are recorded separately.

## Post-main-merge validation

- Merged `main` at `f3db740bc250df087c82d346538a805d4c581ed8`, which includes #5007.
- `uv run gym env test --resources-server partner_if +should_validate_data=true`: 526 passed; data validated.
- `python -m pytest tests/unit_tests/test_partner_if_prepare.py -q`: 6 passed, 1 failed.
  The sole failure is full-release route coverage: only the native Prime row lacks a route.
- Both `file_blocks_v1` and `single_advertised_tool_call_v1` remain registered.
- All 27 published rows validate against the merged schema and match the pinned digest.
- `uv run pre-commit run --all-files`: passed; private guardword catalog unavailable.

No new model rollout was run for this merge; the checker behavior is unchanged and
the earlier six captured rollouts remain the live evidence. The native Prime
integration remains separate; the route-readiness test has not been skipped or relaxed.
