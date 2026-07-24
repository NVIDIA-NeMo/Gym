---
name: nemo-gym-async-patterns-and-performance
description: >-
  Implements and diagnoses NeMo Gym async HTTP, Ray, subprocess, connection-pool,
  and high-concurrency patterns. Use when the user mentions "async pattern",
  "concurrency", "httpx to aiohttp", "performance", "scale", "connection pool",
  "Ray pattern", "high concurrency", event-loop blocking, semaphore sizing,
  sidecar lifecycle, or Ray socket path failures.
---

# Async Patterns and Performance

Use this skill for implementation and diagnosis at concurrency. Preserve asynchronous flow end to end; adding `async def` around blocking work is not sufficient.

Read [Async patterns and performance](../../../fern/versions/latest/pages/infrastructure/async-patterns-and-performance.mdx) before changing clients, workers, or limits.

## Choose the HTTP path

- Inside a Gym server, call another configured Gym server with `self.server_client.post(...)`.
- For lower-level Gym HTTP where no server instance is available, use `nemo_gym.server_utils.request(...)`.
- For OpenAI/Responses API semantics, token metadata, or SDK-style model access, use `NeMoGymAsyncOpenAI` with the Gym aiohttp transport.
- Do not introduce `httpx.AsyncClient` into high-concurrency request paths.
- When a third-party library requires httpx, provide an aiohttp-backed compatibility transport rather than allowing its default pool. Inspect `resources_servers/tavily_search/app.py` for the maintained adapter shape.

## Keep the event loop unblocked

Audit every operation inside async handlers:

- await network and Ray operations directly;
- use `asyncio.create_subprocess_exec` for subprocesses when practical;
- move unavoidable blocking library calls to a bounded executor or isolated worker;
- never call `ray.get()` or synchronous `subprocess.run()` on the event loop;
- avoid CPU-heavy parsing and unbounded file I/O in request handlers.

Ray object refs from async handlers are awaitable. Prefer `result = await remote_fn.remote(...)`; use legacy executor bridges only when the installed Ray/API boundary is demonstrably not awaitable.

## Bound scarce resources, not all requests

Use semaphores around the actual bottleneck: subprocess slots, judge capacity, browser/sandbox instances, or a non-thread-safe dependency. Do not add a global semaphore merely to hide an inefficient client.

Size from measured capacity:

1. determine per-operation CPU, memory, file-descriptor, socket, and downstream limits;
2. start below the first hard limit;
3. load test while observing throughput, latency, queue depth, errors, and memory;
4. increase until throughput plateaus or tail latency/errors rise;
5. retain headroom and document the rationale.

## Advanced process patterns

- Use `multiprocessing.Process` and signal-based hard timeouts only for libraries that cannot be interrupted safely in-process; ensure terminate, join, and cleanup paths run on cancellation.
- Treat sidecars as owned resources: create once at the appropriate server/session scope, perform readiness checks, capture logs, and terminate them on shutdown and failed startup.
- Keep Ray temporary paths short enough for Unix-domain socket limits; diagnose path-length failures before changing networking behavior.
- Use `aiohttp.DummyCookieJar` only when cookie state is intentionally managed elsewhere. Stateful Gym rollouts require correct cookie propagation across seed, tool, and verify calls.

## Diagnose scale failures

Reproduce at increasing concurrency and classify the first saturation point:

- event-loop lag or blocked stacks;
- httpx/httpcore queue growth or pool exhaustion;
- file-descriptor/socket exhaustion;
- downstream 429/5xx/timeouts;
- Ray worker starvation or socket-path errors;
- subprocess/sidecar leaks;
- missing session cookies or state collisions;
- memory growth from retained responses/tasks.

Change one limit or implementation at a time and compare throughput plus tail latency. A lower error rate caused only by serializing all work is not a performance fix.

## Validate

- unit-test cancellation, timeout, and cleanup;
- test malformed downstream responses without leaking sessions;
- run a representative concurrency ramp;
- confirm no unbounded task/process growth after completion;
- report hardware, concurrency, throughput, p50/p95/p99 latency, and error counts.
