# h2-ping-sidecar

Local HTTP/1.1 → HTTPS/HTTP/2 proxy that sends HTTP/2 PING frames while a request is waiting for its first response byte. NVCF endpoints sit behind AWS Global Accelerator, which drops any TCP connection with no application data for 340s; TCP keepalive does not reset that timer, but HTTP/2 PINGs do. Gym's aiohttp client cannot speak HTTP/2, so Gym points its model/judge `base_url` at this sidecar on `127.0.0.1` and the sidecar talks HTTP/2 to NVCF. Credentials (the `Authorization` header) pass through unchanged; the sidecar stores none.

Gym normally starts and stops this process for you from the `sidecar:` config block (see the NVCF sidecar page in the docs). It can also be run by hand:

```bash
go build -buildvcs=false -trimpath -o h2-ping-sidecar .
./h2-ping-sidecar -listen 127.0.0.1:1250 -upstream https://<function-id>.invocation.api.nvcf.nvidia.com
```

| Flag | Default | Meaning |
|---|---|---|
| `-listen` | `127.0.0.1:8080` | Local HTTP/1.1 listen address. Keep it on loopback. |
| `-upstream` | (required) | `https://` base URL to forward to. HTTP/2 only, no HTTP/1.1 fallback. |
| `-ping-interval` | `60s` | Send a PING after this much read-idle time. Must be < 340s. |
| `-ping-timeout` | `15s` | Close the upstream connection if a PING is not acknowledged in time. |
| `-shutdown-grace` | `15m` | On SIGTERM/SIGINT, wait this long for in-flight requests. |
| `-insecure-skip-verify` | `false` | Skip upstream TLS verification (testing only). |
| `-retry-body-limit` | `16777216` | Buffer request bodies up to this many bytes so a request refused with a GOAWAY can be re-sent on a fresh connection. `0` disables buffering. |
| `-ready-file` | (none) | Write the PID here after the listener is bound; removed on normal exit. |

Resource limits come from the Go runtime environment: `GOMEMLIMIT` (for example `1GiB`) and `GOMAXPROCS`.

Build and test need Go 1.27 or newer: `go vet ./... && go test ./...`. The test builds the binary and runs it against a local fake HTTP/2 origin only.

Limitation: the sidecar does not retire long-lived upstream connections on its own schedule. When the load balancer closes one with a GOAWAY, requests whose bodies are within `-retry-body-limit` are re-sent on a fresh connection; larger bodies are not retried.
