# Testing / Validation

Integration testing used a local checkout combining this checker commit (`2962b195f`)
with the changes from [#5007](https://github.com/NVIDIA-NeMo/Gym/pull/5007). The combined
checkout passed all 526 `partner_if` tests. The compatibility merge is not included
in this PR.

## Dataset pin

The registry artifact is now pinned to `partner_if 2.1.5` (27 rows). Upload and download
were verified byte-for-byte against the recorded SHA-256. This branch routes the
new next-call row through the existing text environment. Full-cohort route coverage
remains pending the separately maintained SWE and native Prime integrations; the
route-coverage test is retained as a readiness gate. On the pin update, the component
suite passed 495 tests and data validation. Preparation tests passed 5 tests; the
full-release route-coverage test fails until those integrations land. The 526-test
result above used the combined checkout.
