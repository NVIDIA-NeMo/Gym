# NeMo Fabric Agent

Run interchangeable agent harnesses through [NeMo Fabric](https://github.com/NVIDIA/NeMo-Fabric) inside NeMo Gym.

Install `requirements.txt` in the Gym controller environment. The pinned Hermes runtime requires `openai==2.24.0`, while Gym requires `openai==2.44.0`. Keep its adapter installed in the controller, and install the Hermes runtime in a separate environment. From this directory:

```bash
python3.13 -m venv .hermes-venv
.hermes-venv/bin/python -m pip install -r hermes-runtime-requirements.txt
```

**The Hermes preset requires this extra setup.** Add the child interpreter to its agent configuration:

```yaml
fabric_config:
  environment:
    env:
      ADAPTER_PYTHON: /absolute/path/to/.hermes-venv/bin/python
```

This uses Fabric's existing child-interpreter setting and preserves Gym's dependency pins. The portable helper below builds the Mini/DeepAgents runtime. Package the separate Hermes environment when using the Hermes preset.

The mini-SWE and DeepAgents presets use `model_provider: openai-compatible`, preserving the exact catalog model ID and model-server URL. `runtime-requirements.txt` pins the shared controller dependencies, including Gym's `mcp==1.29.0`. The portable helper resolves these together with Gym so transitive SDK versions stay within Gym's constraints.

That shared installation alone cannot run mini-SWE with LiteLLM 1.106.0. Its tool-call path imports MCP 2 APIs. The dependency helper installs only `mcp==2.3.0` and `mcp-types==2.3.0` with `--no-deps` into `$DEPS_DIR/mini-mcp-site`. Both native pip and cross-platform uv installs use this layout, retaining shared dependency versions. The shared layer must also satisfy MCP 2's requirements, including `httpx2>=2.10.0`, `opentelemetry-api>=1.28.0` and `pydantic>=2.12.0`.

For Mini, the agent finds this site under the existing `NGSWE_AGENT_DEPS_DIR` sandbox runtime path, or `sys.prefix` for a local installation, and prepends it only to the SDK child's `PYTHONPATH`. An authored `fabric_config.environment.env.PYTHONPATH` is retained after that prefix. Without one, the inherited `PYTHONPATH` is retained. Interpreter wrappers selected through Fabric's `ADAPTER_PYTHON` remain unchanged.

For a local venv without the portable dependency helper, create the same layout with:

```bash
python -m pip install --no-deps --upgrade --target "$VIRTUAL_ENV/mini-mcp-site" 'mcp==2.3.0' 'mcp-types==2.3.0'
```

Do not replace MCP 1.29.0 in the Gym controller or export this child path into its process. This extra MCP 2 site is specific to mini-SWE and does not change the DeepAgents preset. If the fixed site is absent, an externally authored child `PYTHONPATH` still works, but shared requirements alone remain insufficient. Include the entire `mini-mcp-site` directory when relocating a runtime bundle. Budgets remain eight turns.

Before considering a runtime ready, exercise native mini-SWE through the actual Gym proxy and Fabric SDK child across at least two model turns. Replay provider thinking blocks and tool-call metadata, check native tool results and submission, and confirm controller and child isolation. A module import or one-call submission cannot establish replay compatibility. Retain original thinking blocks and signatures in native messages and results.

Mini-SWE uses the local `nemo-gym.mini-swe-agent` descriptor and compatibility module. Include both files and Fabric’s installed `share/nemo-fabric/adapters` data in runtime bundles. The existing `nvidia.fabric.mini-swe-agent` alias also selects this adapter. The DeepAgents preset retains its installed descriptor and workspace confinement. To use absolute task paths inside
an isolated task container, add [the Gym DeepAgents descriptor](deepagents.fabric-adapter.json) to
`fabric_config.discovery.local_paths`, select `adapter_id: nemo-gym.deepagents`, and set
`harness_settings.filesystem_root: /`. This changes the filesystem-tool root while preserving `cwd` as the repository.
Keep the default root for host execution.

The mini-SWE preset uses `cost_tracking: ignore_errors` so an unknown catalog price does not abort execution. Unknown costs are reported as `null`. Set `fabric_config.models.default.settings.cost_tracking: default` to retain native pricing errors.

Native model and tool messages are retained in the result. Failed invocations attempt to save the original Fabric result and partial trace to `/logs/agent/fabric-result.json`. An unwritable path logs a warning and preserves the original error or incomplete response. Only Mini's confirmed native `LimitsExceeded` exit returns an `incomplete` response with `stop_reason: turn_limit`, the real partial trajectory, and the failed native result in metadata. This permits normal answer-file verification without creating a final answer. Other failures still raise. Collect the trace file before sandbox cleanup when available.
