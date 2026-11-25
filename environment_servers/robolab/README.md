# RoboLab Environment Server

See [benchmark documentation](../../benchmarks/robolab/README.md) for runtime
inputs, SDK 1.5 / `open_model` arm contracts, sealed terminal auditing, and
evaluation commands.

This Environment Server (`BaseEnvironmentServer`) binds staged single-case or
multi-case (`robolab.efb-suite.v1`) RoboLab manifests, executes the pinned
Bridge runtime adapter, verifies sealed native grade receipts via
`observe_verifier`, and scores model policy failures as unmasked zeros.
