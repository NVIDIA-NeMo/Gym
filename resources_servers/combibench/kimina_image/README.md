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
`nemo_gym.sandbox`. This image needs the same two things plus the REPL binary and the
server, so it follows that Dockerfile's structure rather than upstream Kimina's:

| | upstream Kimina's Dockerfile | this one |
| --- | --- | --- |
| Lean | `elan`, unverified | release tarball, sha256 checked |
| Mathlib | clone at a branch name | pinned commit, and the commit's own `lean-toolchain` asserted to match |
| REPL | `FrederickPu/repl@lean415compat` (for older Leans) | `leanprover-community/repl` at the tag matching Lean, toolchain asserted |
| Final build | online | `--network=none`, so an incomplete cache fails the build |
| Server | — | pinned commit; `uv export --extra server` for the dependency set upstream tests |
| Gate | — | starts a REPL and requires it to load Mathlib and report the version |

That last row is the one that matters most in practice. The server answers `/health` as
soon as FastAPI is up, long before it can start a REPL, so a REPL that cannot start shows
up only as every rollout returning `{"detail":"Failed to start REPL"}`. Building the probe
into the image turns that into a failed build.

If a second REPL-backed Lean benchmark is added, this belongs next to `lean_image` under
`lean_proof/` rather than here.

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

The build downloads the Mathlib cache and takes a few minutes. After start-up the first
proof pays an `import Mathlib` load; later ones reuse the REPL.

`RUN --network=none` is BuildKit syntax. Builders without it (buildah) reject that line;
strip it and the guarantee it gives — that the Mathlib cache is complete offline — is lost,
so re-establish it by running the built image with the network disabled before trusting a
score from it.

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

The flags above are part of the documented command rather than an optional extra: the
container compiles untrusted model output and `native_decide` is allowed (upstream allows
it), so that output can run compiled code in here — the container boundary is the only
isolation there is. Every capability is dropped and a runaway proof cannot fork or
allocate the host to death. Do not run it with `--privileged`, a Docker socket mount, or
host networking.

`--network=none` is not an option — the container exists to answer HTTP on port 8000 — so
the port is published to loopback only, which is the equivalent restriction; on a
multi-tenant host put it on its own Docker network with the Gym resources server instead
of publishing a port at all.

The image does **not** run as root: it creates `lean` (uid 1000) and switches to
it before the last two build steps, so the REPL probe — `lake env repl`, loading
Mathlib and reporting the version — and the server's import check both run as
that user during the build. A successful build is therefore the validation that
the server's own work does not need root. `/opt/mathlib`, `/opt/repl` and
`/opt/kimina` are owned by `lean` because Lake writes under `.lake`; `/opt/lean`
and `/opt/venv` are read-and-execute only. If you mount anything else the REPL
writes to, give `lean` write access to it or the REPL will fail to start.

`--read-only` is not documented above because it is not tested here: the REPL runs as
`lake env` with its working directory inside the Mathlib project and Lake writes there. If
you need it, add `--read-only --tmpfs /tmp` plus a writable mount over
`/opt/mathlib/.lake` and confirm a real `/verify` still succeeds before trusting a score
from it.

Leave `LEAN_SERVER_MAX_REPL_MEM` at the image's **12G**, not 8G. It is applied as
`RLIMIT_AS` on each REPL, and a REPL holding Mathlib exceeds 8G; upstream Kimina raised its
own default to 12G for this reason. At 8G every `/verify` fails with `Failed to start REPL`
while `/health` still answers `ok`, because FastAPI is up long before a REPL is — the build
gate above is what keeps that failure from reaching a run.

Keep the server's `LEAN_SERVER_MAX_REPLS` and the resources server's
`max_concurrent_lean_requests` equal, so the two agree on how many proofs can be in flight.

Converting to squashfs for enroot drops image config, so `ENV` does not survive: pass the
`LEAN_SERVER_*` values explicitly when launching that way. `lake` and `lean` are symlinked
into `/usr/local/bin` precisely because symlinks are filesystem and do survive.

## Provenance

`/opt/image-provenance/` records what was actually built: the Mathlib commit and its
`lean-toolchain` and `lake-manifest.json`, the REPL commit, and the Kimina commit.
