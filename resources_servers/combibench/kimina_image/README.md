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

Upstream CombiBench's own harness verifies through it. Matching its `/verify` contract is
what makes the score here comparable to the published one, which is the only reason this
image exists rather than reusing a sandbox that shells out to `lake env lean`.

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

The container compiles untrusted model output and `native_decide` is allowed
(upstream allows it), so that output can run compiled code in here: the
container boundary is the only isolation. `--network=none` is not an option —
the container exists to answer HTTP — so the port is published to loopback
only. The image still runs as root and has not been validated `--read-only`;
see the security note in the [server README](../README.md#lean-server) for what
that costs and how to go further.

`LEAN_SERVER_MAX_REPL_MEM` defaults to **12G** here, not 8G. It is applied as `RLIMIT_AS`
on each REPL, and a REPL holding Mathlib exceeds 8G; upstream Kimina raised its own default
to 12G for this reason. At 8G every `/verify` fails with `Failed to start REPL`.

Keep the server's `LEAN_SERVER_MAX_REPLS` and the resources server's
`max_concurrent_lean_requests` equal, so the two agree on how many proofs can be in flight.

Converting to squashfs for enroot drops image config, so `ENV` does not survive: pass the
`LEAN_SERVER_*` values explicitly when launching that way. `lake` and `lean` are symlinked
into `/usr/local/bin` precisely because symlinks are filesystem and do survive.

## Provenance

`/opt/image-provenance/` records what was actually built: the Mathlib commit and its
`lean-toolchain` and `lake-manifest.json`, the REPL commit, and the Kimina commit.
