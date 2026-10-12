# Kimina Lean Server image for CombiBench

Lean + Mathlib + the Lean REPL + [Kimina Lean Server](https://github.com/project-numina/kimina-lean-server)
(MIT), built at one pinned Lean version. CombiBench's statements are written against
`v4.24.0`, which is what `versions.py` pins.

## What Kimina is, and why this benchmark needs it

Kimina Lean Server is a FastAPI service that keeps a pool of Lean REPL processes keyed by
import header. `import Mathlib` costs tens of seconds; a REPL that has already loaded that
header is reused, so the cost is paid per pool member rather than per proof. It exposes
`/health` and `/verify`, and starts each REPL as `lake env <repl> ` with the working
directory at the Mathlib project.

## Why not the existing Lean sandbox

Upstream CombiBench's own harness verifies through Kimina, and matching its `/verify`
contract is what makes the score here comparable to the published one. The alternative,
the `math_formal_lean` / `lean_proof` sandbox, shells `lake env lean` through
`nemo_gym.sandbox` once per request:

| | `math_formal_lean` / `lean_proof` sandbox | Kimina Lean Server |
| --- | --- | --- |
| Lean/Mathlib | v4.12.0 and v4.19.0 images | pinned here to v4.24.0, upstream's toolchain |
| REPL reuse | one process per request | header-keyed REPL pool, so `import Mathlib` is paid once |
| Relation to upstream | none | the server upstream's own harness talks to |

The toolchain is the blocking difference: CombiBench's statements do not compile on
v4.12.0 or v4.19.0, so reusing either image would have meant building one at v4.24.0
anyway. The Kimina pin is a commit rather than a release because the project publishes no
tags at all: `fb2393de` (2026-01-11) is the head of `main`, and still the latest commit.
Its own default is Lean v4.26.0; the version is a build argument, so the image here is
built with `LEAN_SERVER_LEAN_VERSION=v4.24.0` to match upstream CombiBench's toolchain.
The consequence of the pin being head-of-branch is that it does not move on its own, and
re-pinning it is a one-line change plus a rebuild.

## Relationship to `lean_proof/lean_image`

`lean_proof/lean_image` builds Lean + Mathlib for the benchmarks that compile through
`nemo_gym.sandbox`. This image needs the same two things plus the REPL binary and the server,
so it follows that Dockerfile's structure rather than upstream Kimina's: the Lean tarball is
sha256-checked, Mathlib and the REPL are pinned to commits whose `lean-toolchain` is asserted
to match, the final build runs with `--network=none` so an incomplete cache fails the build,
and a gate starts a REPL and requires it to load Mathlib and report the version.

That gate matters most in practice: the server answers `/health` as soon as FastAPI is up,
long before it can start a REPL, so a broken REPL otherwise shows up only as every rollout
returning `{"detail":"Failed to start REPL"}`. If a second REPL-backed Lean benchmark is
added, this image belongs next to `lean_image` under `lean_proof/`.

## Build

```bash
python resources_servers/combibench/kimina_image/versions.py  # prints the --build-arg flags below
docker build \
    --build-arg LEAN_VERSION=v4.24.0 \
    --build-arg LEAN_SHA256=b14f5e5159219dd1a1956c3b806813319f5e94ccd5bdfd56f54520609a5bb5ec \
    --build-arg MATHLIB_COMMIT=f897ebcf72cd16f89ab4577d0c826cd14afaafc7 \
    --build-arg REPL_COMMIT=8fff8552292860d349b459d6a811e6915671dc0d \
    --build-arg KIMINA_COMMIT=fb2393de3461db35eda4c714e3fd21187e92ec90 \
    -t kimina-lean-server:v4.24.0 \
    resources_servers/combibench/kimina_image
```

The build downloads the Mathlib cache and takes a few minutes; the first proof after
start-up pays an `import Mathlib` load and later ones reuse the REPL. `RUN --network=none` is
BuildKit syntax: builders without it (buildah) reject that line, and stripping it loses the
guarantee that the cache is complete offline, so run the built image with the network disabled
before trusting a score from it.

## Run

```bash
docker run -d --name kimina-combibench \
    -p 127.0.0.1:12332:8000 \
    --cap-drop=ALL --security-opt=no-new-privileges \
    --pids-limit 512 --memory 32g \
    -e LEAN_SERVER_MAX_REPLS=8 \
    kimina-lean-server:v4.24.0
curl http://127.0.0.1:12332/health     # {"status":"ok"}
```

The flags are part of the documented command: the container compiles untrusted model output
and `native_decide` is allowed, so that output can run compiled code here and the container
boundary is the only isolation. Do not run it with `--privileged`, a Docker socket mount or
host networking. The port is published to loopback only; on a multi-tenant host put it on its
own Docker network with the resources server instead. `--read-only` is untested (Lake writes
under the Mathlib project); if you need it, add a writable mount over `/opt/mathlib/.lake` and
confirm a real `/verify` succeeds first.

The image does not run as root: it creates `lean` (uid 1000), and the build-time REPL probe and
the server's import check both run as that user, so a successful build shows the server does not
need root. Anything else the REPL writes to must be writable by `lean`.

- Leave `LEAN_SERVER_MAX_REPL_MEM` at the image's **12G**. It is applied as `RLIMIT_AS` on each
  REPL, and a REPL holding Mathlib exceeds 8G; at 8G every `/verify` fails with `Failed to start
  REPL` while `/health` still answers `ok`.
- Keep `LEAN_SERVER_MAX_REPLS` equal to the resources server's `max_concurrent_lean_requests`.
  The 8 above is this image's choice, not Kimina's default (`max(cpu_count() - 1, 1)`).
- `LEAN_SERVER_LEAN_VERSION` is set from the `LEAN_VERSION` build arg. It selects nothing (the
  toolchain is the one in `/opt/lean`), but it is what the start-up banner announces; unset,
  Kimina reports its own `v4.26.0`. `LEAN_SERVER_ENVIRONMENT` stays at Kimina's `dev`, since `prod`
  would construct a Google Cloud Logging client, which an offline scoring container should not.
- Converting to squashfs for enroot drops `ENV`, so pass the `LEAN_SERVER_*` values explicitly
  when launching that way. `lake` and `lean` are symlinked into `/usr/local/bin` because
  symlinks survive.

## Provenance

`/opt/image-provenance/` records what was actually built: the Mathlib commit and its
`lean-toolchain` and `lake-manifest.json`, the REPL commit, and the Kimina commit.
