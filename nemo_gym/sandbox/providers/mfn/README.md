# MFN sandbox provider

This provider talks to MFN over async gRPC. It vendors the protobuf messages Gym
needs (`protos/mfn_sandbox.proto`), and its handwritten RPC stub (`rpc.py`) uses
the service's wire-level RPC path prefix, which must match the deployed service.

Install Gym with the `mfn` extra (`pip install "nemo-gym[mfn]"` or `uv sync --extra mfn`; `all` includes it, the
`sandbox` extra does not), set `SANDBOX_SERVICE_ADDRESS` to a
plain-text MFN gRPC `host:port`, and include `configs/mfn.yaml` in the Gym
configuration paths.

Supported operations are create/readiness, exec, single-file upload/download,
status, shutdown, declared HTTP endpoints, reconnect-by-ID, and duplex PTY/pipe
sessions. PTY dimensions can be changed during an active session on current
MFN deployments. Older deployments ignore resize frames and leave the previous
dimensions unchanged. Attaching from another client, detached PTY execution,
and signals other than terminal `SIGINT` are not supported.

Noninteractive commands and the creation probe use `ExecStream` and immediately
half-close its stdin side. This also works with local MFN backends whose
legacy `Exec` RPC keeps its separate stdin channel open.

MFN-specific creation options belong in `SandboxSpec.provider_options`:

```yaml
sandbox_spec:
  provider_options:
    snapshot_id: null  # Mutually exclusive with sandbox_spec.image.
    require_vm: false
    beta_no_ttl_cap: false
    shard_pin: null
    network_mode: allow_all  # null, allow, block, or allow_all
    allowed_cidrs: []
    allowed_domains: []
    blocked_cidrs: []
    blocked_domains: []
    network_name: ""
```

After changing `protos/mfn_sandbox.proto`, regenerate its message bindings:

```bash
nemo_gym/sandbox/providers/mfn/protos/generate.sh
```

The handwritten `rpc.py` stub intentionally exposes only the RPCs Gym uses.
