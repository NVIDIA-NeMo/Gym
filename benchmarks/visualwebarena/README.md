# VisualWebArena

This benchmark uses BrowserGym's VisualWebArena task and evaluator with a
shared Gym rollout loop. Observations contain the screenshot with BrowserGym's
set-of-marks overlay plus a bracketed accessibility tree.

The upstream repository stores site-local task IDs in three files, while
`libvisualwebarena==0.0.15` and BrowserGym use one global range `0..909`. The
prepare script reproduces the package order exactly: Classifieds, Reddit
(including its cross-site tasks), then Shopping, assigning global IDs after
concatenation. For example, upstream Shopping task 0 is BrowserGym task 444.

The default source is the sibling `../visualwebarena/config_files/vwa`
checkout. Set `VISUALWEBARENA_SOURCE_DIR` for another layout, then run:

```bash
gym eval prepare --benchmark visualwebarena
```

Deploy the VWA site stack and configure the required `VWA_*` variables before
collecting rollouts.
