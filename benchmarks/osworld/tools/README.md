# OSWorld public tools

Each first-time deployment check and runtime step has one checked-in entry
point. `prepare.py` remains at the benchmark root because it is the benchmark
configuration entry point; host checks and lifecycle wrappers live here:

```text
model host       -> probe_model_endpoint.py
environment host -> check_environment.sh
agent/control     -> prepare.py -> prefetch/opt-in deps -> start_control.sh -> run_eval.sh
abnormal recovery -> cleanup_run.sh -> cleanup_opensandbox_run.py
```

| Tool | Purpose |
| --- | --- |
| `probe_model_endpoint.py` | Require the configured model identity and optionally exercise the one- or three-image chat-completions request shape |
| `super35_reasoning.py` | Optional vLLM 0.32.0 plugin preserving Super 3.5 action content after the first thinking boundary |
| `check_super35_reasoning.py` | CPU regression checks against the real pinned vLLM parser and checkpoint tokenizer |
| `check_environment.sh` | Validate local or SSH-reached Linux/Docker/KVM/qcow2 environment-host readiness |
| `start_control.sh` | Preflight the build toolchain and explicitly prepared OSWorld agent venv, then run `gym env start` |
| `run_eval.sh` | Supervisor-friendly wrapper around `gym eval run --no-serve` |
| `cleanup_run.sh` | Recovery-only cleanup for stale processes or labeled Sandbox containers after abnormal termination |
| `cleanup_opensandbox_run.py` | Read-only audit or opt-in reaping of OpenSandbox instances matching one exact run ID |
| `prepare_osworld_vm.sh` | Download and verify the pinned OSWorld qcow2 baseline |

Model serving itself belongs to the selected model's deployment project;
`probe_model_endpoint.py` is the provider-neutral contract check used before
Gym starts. For example:

```bash
python3 benchmarks/osworld/tools/probe_model_endpoint.py \
  --base-url http://MODEL_HOST:8000/v1 \
  --api-key local-vllm \
  --model SERVED_MODEL_NAME \
  --image-count 3
```

## Super 3.5 reasoning boundary

The optional `super35_reasoning` plugin runs in the **vLLM serving environment**.
Gym does not import it or acquire a vLLM dependency. It is qualified against
vLLM **0.32.0** and the public checkpoint
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` at revision
`fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe`.

An actual response with two `</think>` delimiters exposed a serving-integrity
problem: the native `nemotron_v3` engine omitted the second delimiter from
returned action content. The run's raw-generation guard stopped before that
action executed. This plugin splits at the **first** closing delimiter and
preserves the entire remaining suffix, including any extra tags or malformed
JSON, XML, or Python. Gym's action parser still decides whether that content is
valid. A missing thinking boundary produces no action content; it is not repaired.

This is a non-streaming, thinking-enabled protocol. Streaming, disabled
thinking, and `force_nonempty_content` are rejected. Reasoning token counting
is inherited from the native parser. The pre-existing reasoning-only `rstrip()`
presentation rule is retained: this is **not** a byte-preservation claim for the
reasoning field. Keep original completion token IDs and decoded generations
when auditing either field. The plugin itself does not collect those traces.

Run the CPU checks inside the pinned serving environment with the checkpoint
already downloaded to `MODEL`. The checks use real vLLM classes and the real
tokenizer without loading weights. They cover the original repeated-delimiter
response, ordinary responses, whitespace, missing delimiters, malformed JSON
and Python, repeated opening/closing tags, empty fields, and unsupported modes.
The report retains every case's original text, token IDs, and both parser results.

```bash
export MODEL=/absolute/path/to/pinned-super35-checkpoint
export GYM_ROOT=/absolute/path/to/Gym
export PYTHONPATH="$GYM_ROOT/benchmarks/osworld/tools${PYTHONPATH:+:$PYTHONPATH}"

python "$GYM_ROOT/benchmarks/osworld/tools/check_super35_reasoning.py" \
  --tokenizer "$MODEL" \
  --output /absolute/path/to/new-reasoning-check.json
```

Load the plugin **by module name**, as below. vLLM loads plugins at multiple
startup stages; Python caches a module-name import, which the check also tests.
Loading the same file by path repeatedly can attempt duplicate registration.
The following server settings were used for the 128-worker, fixed-three-image,
100,000-output-token experiment (R6); the serving image limit remains four:

```bash
vllm serve "$MODEL" \
  --served-model-name nemotron-3.5-super-vl \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --allowed-local-media-path / \
  --max-model-len 262144 \
  --max-num-seqs 128 \
  --limit-mm-per-prompt '{"video": 1, "image": 4}' \
  --video-pruning-rate 0.5 \
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
  --seed 0 \
  --reasoning-parser-plugin super35_reasoning \
  --reasoning-parser nemotron_v3_lossless_nonstream
```

MTP was disabled. Omit both `--enable-auto-tool-choice` and
`--tool-call-parser`; `qwen3_coder` can recover malformed calls or remove raw
XML before Gym sees it. The protocol sends API tool schemas with
`tool_choice: none`, `skip_special_tokens: false`, and `stream: false`.
The model's generation prompt already opens `<think>`; this splitter must not
be applied to an unrelated prompt or reasoning protocol.

The R6 rollout used NeMo Gym commit
`462d5f971b08f165eb98ceb511ba2d9201b2976a`, protocol
`nemotron-super35-tool-xml-v3`, and unchanged action parser
`nemotron-super35-tool-xml-parser-v1`. Its original plugin SHA-256 is
`9f6f70809eae7dee558b95b8ead7d2cae65471d02251052e4ab00cc1604ea4b6`.
This checked-in copy adds documentation, SPDX metadata, and type annotations;
the extraction method bodies and registration are unchanged. The module-name
loading recipe is a packaging improvement validated separately from that run.

The full run used the registered runner and `run_osworld_task` with Gym's
message-building client and a direct vLLM endpoint. It did not exercise
`prepare.py` plus the complete Gym server stack. Its agent settings were:

```yaml
runner_name: nemotron_super35_agent
model_protocol_id: nemotron-super35-tool-xml-v3
history_policy:
  name: fixed
  params: {keep_images: 3}
max_steps: 100
max_tokens: 100000
temperature: 0.6
top_p: 0.95
sleep_after_execution: 5.0
task_timeout: 14400
agent_kwargs:
  coordinate_type: relative
  thinking: true
  parse_retries: 5
  parse_error_feedback: true
  parse_retry_temperature: null
  pre_done_checklist: false
  repeated_action_warning_threshold: 0
  repeated_action_window: 12
```

`keep_images: 3` includes the current screenshot and does not pin the initial
screenshot. Text and thinking history remain present. The controller used
two 64-worker shards, a 3,600-second model timeout and 18,000-second desktop TTL.
The checkpoint ran on four GB200 GPUs with tensor parallelism four, while
desktops used the Cell-3 guest image with SHA-256
`a5118c5fd065fc8b5bbd3a739b278a40cd3648f566f9a1d330c96293f2844638`.
OSWorld was pinned to `f32ab2b74e3ea66e6a8eb0d87876a12ce93904d5`.
These are experiment settings, not replacements for the default Super 3.5
profile. To select fixed history in a new local profile, replace its whole
`history_policy` mapping; merging `keep_images` into `sink_window.params`
leaves incompatible keys and is rejected.

The task subset, VM image, source versions, retry budget, output budget,
sampling, serving parser and history policy all belong to a run's identity.
Freeze them before starting, verify raw-generation/content parity, and retain
malformed model responses. Do not hot-patch a running experiment or reuse an
earlier score for a changed configuration. Full results and comparison limits
are recorded in [PR #5101](https://github.com/NVIDIA-NeMo/Gym/pull/5101).

## Environment lifecycle

Validate an environment host locally, or stream the same checked-in checker to
a remote host that does not have a Gym checkout:

```bash
# On the environment host:
bash benchmarks/osworld/tools/check_environment.sh /absolute/path/to/Ubuntu.qcow2

# Or, from its paired agent/control host:
bash benchmarks/osworld/tools/check_environment.sh \
  --ssh REMOTE_USER@ENV_HOST_REACHABLE_IP \
  /same/absolute/path/on/both/hosts/Ubuntu.qcow2
```

The checker pins the public OSWorld disk size and SHA-256 by default. Set
`EXPECTED_VM_SHA256`, `EXPECTED_VM_SIZE`, or `MIN_FREE_GIB` only when the
deployment intentionally uses a different verified baseline or capacity
threshold.

Both runtime wrappers require `OSWORLD_RUN_ID`. Set
`NEMO_GYM_CONTROL_HOST` when the control services must advertise a non-loopback
address. Their optional positional argument selects the root for logs and
results; it defaults to the Gym repository root.

`prepare.py` prints the exact `gym env prefetch` and
`install_optional_runtime_deps.sh` commands for the configured managed agent
venv. The install remains an explicit opt-in because these packages are
excluded from Gym's shipped environments. `start_control.sh` validates their
versions and imports and prints the same remediation if they are not ready; it
does not install anything automatically.

For a split-host Gym Docker deployment, run the wrappers on the agent/control
host and point its normal Docker CLI at the OSWorld environment host. The
identical qcow2 path and non-interactive SSH authorization are one-time host
preparation; they are not repeated for each benchmark run:

```bash
export DOCKER_HOST=ssh://REMOTE_USER@ENV_HOST_REACHABLE_IP
export OSWORLD_SANDBOX_PUBLISH_HOST=ENV_HOST_REACHABLE_IP
bash benchmarks/osworld/tools/check_environment.sh \
  --ssh REMOTE_USER@ENV_HOST_REACHABLE_IP \
  /same/absolute/path/on/both/hosts/Ubuntu.qcow2
docker info  # validates Docker's SSH transport from this host
```

`start_control.sh` revalidates both variables and the Docker connection before
starting Gym. The qcow2 path written by `prepare.py --vm-path` must exist at
the same absolute path on the Docker host. Once `docker info` succeeds, each
run uses only `prepare.py`, `start_control.sh`, and `run_eval.sh`. Export the
same `DOCKER_HOST` when running `cleanup_run.sh` so cleanup remains scoped to
the correct daemon.

Only when normal termination fails or stale run-owned entities block recovery,
stop that run while preserving its logs and results:

```bash
export OSWORLD_RUN_ID=my-osworld-run
bash benchmarks/osworld/tools/cleanup_run.sh /absolute/run/root
```

The runtime wrappers record PIDs under `RUN_ROOT/run/osworld/RUN_ID/`. Cleanup
validates both the PID environment and command before signaling it. Docker
cleanup requires the Sandbox, OSWorld workload, and run-ID labels to match; it
does not remove unlabeled or other-run containers. Model services are outside
this lifecycle and are never stopped by this tool.

When both `OPENSANDBOX_BASE_URL` and `OPENSANDBOX_API_KEY` are set,
`cleanup_run.sh` also invokes the OpenSandbox reaper through the Gym Python
environment. The reaper queries both OSWorld's `run-id` metadata and Gym's
`nemo-gym.nvidia.com/run` attribution metadata, then rechecks returned metadata
client-side before terminating anything. Run it without `--reap` for a
read-only audit:

```bash
export OSWORLD_RUN_ID=my-osworld-run
.venv/bin/python benchmarks/osworld/tools/cleanup_opensandbox_run.py
```
