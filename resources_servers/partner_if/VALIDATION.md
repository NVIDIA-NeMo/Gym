# Testing / Validation

## Current-main conflict resolution

Merged `main` at `bd6f5f0d8f3a56cbf3f7e1d2d4a1d2220b7ff6fe` into the PR branch.
Kept the managed `pyproject.toml` / `uv.lock` dependency format, preserving Prime's
`sandbox` extra, and updated the Prime adapter to the current public imports and
`ResourcesSessionFences` API. The checker, dataset pin, and task routes are unchanged.

Fresh checks against the resolved implementation:

```bash
gym env test --resources-server partner_if +should_validate_data=true +skip_venv_if_present=true
# 561 passed; example data validation passed

gym env test +entrypoint=responses_api_agents/prime_agent +should_validate_data=true +skip_venv_if_present=true
# 67 passed

python -m pytest tests/unit_tests/benchmarks/test_partner_if_prepare.py tests/unit_tests/test_public_api_facades.py -q
# 844 passed, including the complete 27-task route-coverage check

uv lock --project responses_api_agents/prime_agent --check
# Passed with uv 0.13.4
```

Component tests reused the existing local virtual environments. No new model or
cluster rollout was run: these fixes preserve the existing adapter behavior while
following upstream packaging and API changes. Earlier rollout evidence remains
applicable to the unchanged checker; it is not new cluster validation.

The sections below record earlier integration stages and their then-current gaps.

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
