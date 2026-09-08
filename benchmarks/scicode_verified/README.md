# SciCode-Verified

Native NeMo Gym integration for [SciCode-Verified](https://github.com/flyingwagner/scicode-verified),
the human-verified v2 correction of the SciCode scientific code-generation benchmark.

- **Release**: 64 main problems, 290 total subproblems, and 287 scored subproblems
- **Reward**: `1.0` only when every scored subproblem in a main problem passes
- **Metrics**: whole-problem accuracy (`mean/reward`) and subproblem-weighted `subtask_accuracy`
- **Protocol**: pass@1; report with and without expert background separately; 1,800-second timeout
  per subproblem and grading environment; accept a subproblem if it passes in either the pinned
  2024-era or 2025-era scientific-Python environment

This benchmark reuses Gym's native `scicode_agent` cumulative generation loop and `scicode`
resources server. It does not wrap SciCode-Verified's provider-specific evaluation launcher.

## Pinned release and integrity

Preparation is pinned to:

- SciCode-Verified source commit `ddab4a92f8d80a7113ab946628e994b52354d838`
- Hugging Face dataset revision `eea11a866be6860725258702b39ef8651ed26abd`
- `problems_test.jsonl` MD5 `5c604d8dbf52642bd94e13b92c8f52eb`
- `test_data_cleaned.h5` MD5 `2b41a7df40ddc23ce651ec05b8ecb6f8`

The preparation script downloads the released JSONL, manifest, corrected approximately 1.1 GB
HDF5, and the three official unscored-step implementations. It verifies every pinned hash before
writing Gym data. The resources server checks the HDF5 again before grading.

```bash
gym eval prepare --benchmark scicode_verified
```

Generated data is placed under `benchmarks/scicode_verified/data/` and is not committed.

## Grading interpreters

The official evaluator accepts a subproblem if it passes under either of two explicitly supplied
Python interpreters. Configure those same environments in an `env.yaml` overlay:

```yaml
scicode_verified_resources_server:
  resources_servers:
    scicode:
      grading_interpreters:
        - name: "2024"
          python_executable: /path/to/scientific-python-2024/bin/python
        - name: "2025"
          python_executable: /path/to/scientific-python-2025/bin/python
```

Each environment must contain the packages imported by the SciCode problems and verifier,
including NumPy, SciPy, h5py, matplotlib, SymPy, and NetworkX. The upstream release describes the
stacks as the NumPy 1.26/SciPy 1.13 era and the 2025 current era, but does not distribute complete
environment lockfiles. Use the exact interpreter paths from the run being reproduced and record
their package inventories with the result.

The benchmark requires two resolved interpreters and fails with a clear error otherwise. The
original `scicode` benchmark remains backward-compatible and continues to use its resources
server's current interpreter unless `grading_interpreters` is configured.

## Run

Run the with-background condition:

```bash
gym eval run \
  --model-type vllm_model \
  --benchmark scicode_verified \
  --split benchmark \
  --output results/scicode_verified_with_background.jsonl \
  ++reuse_existing_data_preparation=true
```

Run the without-background condition separately:

```bash
gym eval run \
  --model-type vllm_model \
  --benchmark scicode_verified/configs/no_background \
  --split benchmark \
  --output results/scicode_verified_without_background.jsonl \
  ++reuse_existing_data_preparation=true
```

Do not merge the two conditions when reporting results. The canonical headline values are pass@1
whole-problem accuracy over 64 problems and subtask accuracy over 287 scored subproblems.

## Official skipped steps

Subproblems `13.6`, `62.1`, and `76.3` are not generated or scored. Preparation downloads their
official reference implementations from the pinned SciCode-Verified source revision and embeds
them in the corresponding task rows. The agent adds that code to cumulative context for later
subproblems. This benchmark-specific field takes precedence over the original SciCode fallback
constants without changing the existing benchmark.

## Validation before baselining

Before setting the resources server's `verified` metadata to true, require:

1. A one-problem real-model smoke and a smoke containing an official skipped step.
2. All reference solutions passing all 287 scored subproblems.
3. Cached generations regraded by the official evaluator and Gym with exact per-subproblem parity.
4. Separate with- and without-background pass@1 reports.

## Licensing and attribution

Gym code and SciCode-Verified data are Apache 2.0. Cite both the original SciCode benchmark and
SciCode-Verified, as requested by the release authors.
