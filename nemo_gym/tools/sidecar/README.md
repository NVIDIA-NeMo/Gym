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
| `-max-conn-age` | `50m` | Retire the upstream connection after 90-100% of this age: in-flight requests finish on it and new requests use a fresh connection. Keep it below the upstream's client keep-alive limit (AWS ALB default 3600s). `0` disables. |
| `-ready-file` | (none) | Write the PID here after the listener is bound; removed on normal exit. |

Resource limits come from the Go runtime environment: `GOMEMLIMIT` (for example `1GiB`) and `GOMAXPROCS`.

Build and test need Go 1.25 or newer: `go vet ./... && go test ./...`. The tests build the binary and run it against local fake HTTP/2 origins and relays only. They are not run in CI.

Gym builds this program on first use and caches it under a path that contains a hash of the sources and of the Go version (`go env GOVERSION`), so editing the sources, or upgrading Go (whose `net/http` and `crypto/tls` this proxy mostly is, so security fixes ship as Go releases), triggers a rebuild. The minimum Go version in `go.mod` is owned by whoever changes this directory: raise it only when the code needs it, and note that `pyproject.toml`'s dependency policy covers Python packages only. 1.25 is the practical floor because the Go runtime follows a container's CPU quota automatically from that release.

Connection lifetime: the load balancer closes a client connection with a GOAWAY once it is about an hour old. By default the sidecar retires its upstream connection after 45-50 minutes (`-max-conn-age`); requests in flight finish on the old connection and new ones use a fresh one. A request still running when the limit is reached can meet the GOAWAY: bodies within `-retry-body-limit` are re-sent on a fresh connection, larger ones are not retried.
