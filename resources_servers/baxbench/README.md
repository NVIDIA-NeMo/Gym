# BaxBench

[BaxBench](https://github.com/logic-star-ai/baxbench) ([paper](https://arxiv.org/abs/2502.11844), MIT) asks a
model to write a complete backend from an OpenAPI spec: 28 scenarios in 14 frameworks and 6 languages, 392
tasks. Upstream's functional tests check that the app works, and its expert-written exploits and crash
monitors check that it is secure. The reward is upstream's secure pass: every functional test passes and no
exploit finds a CWE. `baxbench/sec_pass_at_1` (higher is better) is the headline metric, next to
`baxbench/pass_at_1` and `baxbench/insecure_among_correct`.

Everything comes from upstream at commit `de885cd9` (`upstream.py`): the prompts (`benchmarks/baxbench`),
the test and exploit code, and the framework Dockerfiles. Nothing is vendored. The answer parser
(`parsing.py`) reproduces upstream's `Parser`, because upstream's module imports the model clients.

## How a sample is graded

Upstream builds a Docker image per sample and starts one container per test. This server needs no
Docker daemon. It starts one OpenSandbox sandbox per test from the prebuilt image of the framework. In
the sandbox, `sandbox/sandbox_runner.py` does the following:

1. It copies the generated files with the Dockerfile's code `COPY` and runs the Dockerfile steps after it
   (`dockerfile_split.py`).
2. It starts the entrypoint, waits for port 5000, and runs one upstream test function unchanged.
3. `docker_shim.py` serves upstream's Docker calls (exec, archives, memory stats) from the local process
   table and filesystem.
4. It applies upstream's crash rule: a server that dies during a test adds CWE-703.

The tests run inside the sandbox next to the app, as upstream runs them next to the container. They reach
the app on `localhost`, without a proxy that could change exploit requests.

The server applies upstream's scoring:

- An image that does not build fails every test.
- An exploit that errors out counts as secure (upstream issue #18). Such errors are counted in
  `baxbench/samples_with_st_exceptions`.
- An answer with no code block fails without a sandbox.
- If a sandbox fails or the runner crashes, the test is retried `inconclusive_verification_retries`
  times. If the test still fails, the sample is masked (`mask_sample`, `failure_kind`).

## Framework images

`sandbox/images/<framework>/` holds one build context per framework. `sandbox/images/render.py` writes
them from upstream:

- the framework Dockerfile up to the code `COPY`
- upstream's common packages and the extra packages of every scenario for that language
- a harness layer under `/opt/baxbench` (Python 3.12, the test dependencies and the upstream source) that
  is not on the app's `PATH`

Three lines are pinned for reproducibility: a Rust 1.83-compatible `Cargo.lock`, Rails 8.0.4 (upstream's
Gemfile asks for `~> 8.0.1`) and the Lumen 10.0.0 skeleton. The Rails image also installs upstream's
Gemfile, which replaces the generated one in every sample, so the per-sample `bundler install` needs no
network. Upstream's Lumen `composer.json` is not valid JSON, so its per-sample `composer install` always
fails, hidden by `|| echo`, and the app runs on the vendor directory of the skeleton, here as upstream.
Build and push each image, then set the template (`{env}` is the directory name):

```bash
python -m resources_servers.baxbench.sandbox.images.render
for d in resources_servers/baxbench/sandbox/images/*/; do
  docker buildx build --platform linux/amd64,linux/arm64 --push \
    -t "<registry>/baxbench-$(basename "$d"):de885cd9" "$d"
done
export BAXBENCH_IMAGE_TEMPLATE='<registry>/baxbench-{env}:de885cd9'
```

## Differences from upstream

- Each image has the extra packages of every scenario of its language. Upstream installs only the
  packages of the scenario under test.
- Grading has no network by default (`network_policy`). Upstream builds with network access. A sample
  that installs a package outside its allow list can therefore fail here: for example, a new gem in a
  Rails `Gemfile` or a module that `go mod tidy` must fetch.
- The 1 GiB container memory limit is enforced by a watchdog on the app's resident memory. Upstream uses
  the cgroup limit, which also counts the page cache.
- The app runs as root in the same sandbox as the test, where upstream keeps the harness outside the
  container. Every process the build or the app starts is stopped before the result is written; an app
  written to attack the grader could still interfere while its test runs.
- Upstream's wait for the server start has no request timeout. Here a server that accepts connections but
  never answers fails the test after 60 s, and does not stall the run.
- The test code is unchanged, with its known bugs: the ZipToTxt exploit never runs (upstream PR #22), and
  UptimeService has a duplicate `serviceId` (upstream PR #19).

## Running

```bash
gym eval prepare --benchmark baxbench
BAXBENCH_IMAGE_TEMPLATE='<registry>/baxbench-{env}:de885cd9' gym eval run \
    --model-type vllm_model --benchmark baxbench --output results/baxbench.jsonl
```

This needs `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` (the provider config in
`nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml`). One sample uses 2 to 6 sandboxes
(2 CPUs and 4 GiB each), and `max_concurrent_sandboxes` caps them.
