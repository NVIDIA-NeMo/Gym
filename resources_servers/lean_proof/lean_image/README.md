# Lean/Mathlib sandbox image

One image per Mathlib version. Compiled `.olean` files do not carry across versions, so a
benchmark must run against exactly the version its statements were written for. The wrong one
fails in a way that looks like model error: measured on LeanCat against Mathlib v4.12.0, 36 of
its 100 reference statements fail to compile **with their `sorry` still intact**, so they score
0 whatever the model writes — a silent cap at 64/100, skewed by difficulty.

`versions.py` holds the pins; `build.sh` takes the version.

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

Per version, `versions.py` holds the Lean release tarball's sha256 and the commit its Mathlib
tag resolves to. The build:

- verifies the tarball checksum before unpacking;
- fetches Mathlib at that exact commit, then asserts the commit's own `lean-toolchain` names
  the Lean version being installed, so a mismatched pair fails the build rather than producing
  a subtly wrong image;
- runs the final `lake build Mathlib` with `--network=none`, so an incomplete cache fails here
  instead of during a rollout;
- compiles a Mathlib-importing file under `env -i`, which is how the verifier invokes it.

Provenance is kept in `/opt/image-provenance/` inside the image: the Mathlib commit,
`lean-toolchain` and `lake-manifest.json`.

## Adding a version

See the instructions in `versions.py`. One entry there is all a new Mathlib version needs.

## Layout

- `/opt/lean` — toolchain, also symlinked into `/usr/local/bin`
- `/opt/mathlib` — the Mathlib project, prebuilt; the working directory for `lake env lean`
- `CMD ["sleep", "infinity"]` — the sandbox injects its own exec daemon, so the image needs
  only a process that stays alive, not a server.
