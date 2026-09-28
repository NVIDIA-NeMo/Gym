# Lean/Mathlib sandbox image

One image per Mathlib version, for the Gym Lean benchmarks. Compiled `.olean` files do not
carry across versions, so a benchmark run against the wrong Mathlib fails in ways that look
like model errors: on Mathlib v4.12.0, 36 of leancat's 100 reference statements fail to
compile **with their `sorry` still intact**, which caps any score at 64/100 and skews it by
difficulty (Easy 4, Medium 11, High 21 unusable). The version has to be exact.

| Version | Benchmarks |
|---|---|
| `v4.12.0` | minif2f, proofnet, putnam_bench, mobench |
| `v4.19.0` | leancat |
| `v4.33.0`, `v4.34.0` | pinned and ready, unused here so far |

`v4.34.0` matches the Lean layer of
`responses_api_agents/opencode_sandboxed_agent/offline_science_image`, whose pinning approach
this follows.

## Build

```bash
./build.sh v4.19.0                      # local: gym-lean:v4.19.0
./build.sh v4.19.0 <registry>/gym-lean  # build and push, prints the digest
```

Pin the **digest** in benchmark configs, not the tag: a tag can be moved, a digest cannot.

```yaml
sandbox_config:
  image: <registry>/gym-lean@sha256:<digest>
```

## What is pinned, and why it fails loudly

`versions.py` holds, per version, the Lean release tarball's sha256 and the commit its
Mathlib tag resolves to. The build:

- verifies the tarball checksum before unpacking;
- fetches Mathlib at that exact commit, then asserts the commit's own `lean-toolchain` names
  the Lean version being installed, so a mismatched pair fails the build rather than
  producing a subtly wrong image;
- runs the final `lake build Mathlib` with `--network=none`, so an incomplete cache fails
  here instead of during a rollout;
- asserts `lake` and `lean` resolve under `env -i`. The verifier execs them through a
  non-login shell that reads no profile, and an image that only works under a login shell
  fails every task with `lake: not found`.

Provenance is kept in `/opt/image-provenance/` inside the image: the Mathlib commit,
`lean-toolchain` and `lake-manifest.json`.

## Adding a version

```bash
git ls-remote https://github.com/leanprover-community/mathlib4 refs/tags/<tag>
curl -fsSL https://github.com/leanprover/lean4/releases/download/<tag>/lean-<v>-linux.tar.zst | sha256sum
```

Add the pair to `versions.py`. Mathlib releases name the Lean toolchain they require, so
take the Lean version from the Mathlib tag rather than choosing it.

## Layout the servers expect

- `/opt/lean` — toolchain, on `PATH`
- `/opt/mathlib` — the Mathlib project, prebuilt; the working directory for `lake env lean`
- `CMD ["sleep", "infinity"]` — OpenSandbox injects its own exec daemon, so the image needs
  only a process that stays alive, not a server.
