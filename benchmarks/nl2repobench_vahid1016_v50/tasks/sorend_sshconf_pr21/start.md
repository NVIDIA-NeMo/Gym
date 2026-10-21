## sshconf Project Introduction and Goals

`sshconf` is a lightweight, pure-Python library for reading and modifying OpenSSH client configuration files (`~/.ssh/config` and files referenced by `Include` directives) in a non-intrusive way: after modifications the file should look roughly the same as before, preserving original lines, comments, blank lines and indentation where possible. The project is a small, single-module library originally authored at the Department of Computer Science, Sri Venkateswara University, Tirupati, India, and distributed on PyPI. It supports parsing single config files, merging configurations spread over multiple files via `Include`, querying and editing host settings, adding/renaming/removing host entries, and writing the result back either to the original files (`save()`) or to a new file (`write()`).

## Natural Language Instructions (Prompt)

1. Implement the complete `sshconf` library from scratch as an importable Python package. The public surface must live in a top-level module importable as `import sshconf`, with an `__version__` attribute (default `"0.0.dev0"`; a CI versioning step may replace it). The library has no third-party runtime dependencies — use only the Python standard library (`os`, `re`, `glob`, `collections.defaultdict`, `collections.Counter`).

2. Include the full list of known ssh parameters (from `man ssh`), such as `AddKeysToAgent`, `AddressFamily`, `BatchMode`, `BindAddress`, `CanonicalDomains`, `CanonicalizeFallbackLocal`, `CanonicalizeHostname`, `CanonicalizeMaxDots`, `CanonicalizePermittedCNAMEs`, `CertificateFile`, `ChallengeResponseAuthentication`, `CheckHostIP`, `Cipher`, `Ciphers`, `ClearAllForwardings`, `Compression`, `CompressionLevel`, `ConnectionAttempts`, `ConnectTimeout`, `ControlMaster`, `ControlPath`, `ControlPersist`, `DynamicForward`, `EscapeChar`, `ExitOnForwardFailure`, `FingerprintHash`, `ForwardAgent`, `ForwardX11`, `ForwardX11Timeout`, `ForwardX11Trusted`, `GatewayPorts`, `GlobalKnownHostsFile`, `GSSAPIAuthentication`, `GSSAPIKeyExchange`, `GSSAPIClientIdentity`, `GSSAPIDelegateCredentials`, `GSSAPIRenewalForcesRekey`, `GSSAPITrustDns`, `GSSAPIKexAlgorithms`, `HashKnownHosts`, `Host`, `HostbasedAuthentication`, `HostbasedKeyTypes`, `HostKeyAlgorithms`, `HostKeyAlias`, `HostName`, `IdentitiesOnly`, `IdentityAgent`, `IdentityFile`, `Include`, `IPQoS`, `KbdInteractiveAuthentication`, `KbdInteractiveDevices`, `KexAlgorithms`, `LocalCommand`, `LocalForward`, `LogLevel`, `MACs`, `Match`, `NoHostAuthenticationForLocalhost`, `NumberOfPasswordPrompts`, `PasswordAuthentication`, `PermitLocalCommand`, `PKCS11Provider`, `Port`, `PreferredAuthentications`, `Protocol`, `ProxyCommand`, `ProxyJump`, `ProxyUseFdpass`, `PubkeyAcceptedKeyTypes`, `PubkeyAuthentication`, `RekeyLimit`, `RemoteForward`, `RequestTTY`, `RhostsRSAAuthentication`, `RSAAuthentication`, `SendEnv`, `ServerAliveInterval`, `ServerAliveCountMax`, `StreamLocalBindMask`, `StreamLocalBindUnlink`, `StrictHostKeyChecking`, `TCPKeepAlive`, `Tunnel`, `TunnelDevice`, `UpdateHostKeys`, `UsePrivilegedPort`, `User`, `UserKnownHostsFile`, `VerifyHostKeyDNS`, `VisualHostKey`, and `XAuthLocation`. Expose `KNOWN_PARAMS` as an uppercase tuple and `known_params` as the lowercase list.

3. Provide a `ConfigLine` class that stores `line`, `host`, `key`, `value` (all defaulting to `None` except `line`), with a `__repr__` describing the fields.

4. Provide single-file operations:
   - `read_ssh_config_file(path)` — opens the file, splits into lines (dropping trailing newlines only), and returns a parsed config-file object.
   - `empty_ssh_config_file()` — returns an empty configuration.

5. Implement an internal line model:
   - Key/value parsing splits on the first whitespace run after stripping inline `#` comments; lines with fewer than two parts are stored as bare `ConfigLine(line=...)` entries (comments, blank lines) with no host/key/value tracking beyond current-host association.
   - `key.lower() == "host"` lines start a new host block; subsequent keyed lines belong to that host until the next `Host` line. Host values may be patterns like `*`; all are collected into a host set.
   - Indentation is auto-probed: the most frequently used leading whitespace of keyed (non-Host) lines becomes the file's indent for newly generated lines; default two spaces.

6. Implement a `SshConfigFile`-equivalent class (constructed from a list of lines) exposing:
   - `hosts()` → tuple of host entries found (may include `"*"`); host order is set-based.
   - `host(host)` → dict keyed by lowercase parameter names, excluding `Host` itself; when a parameter appears once the value is the plain string, when multiple times it is the list of values; empty dict if the host is unknown.
   - `set(host, **kwargs)` — overwrite existing settings and/or add new ones; single values and lists/tuples of values are supported; when a list has fewer items than existing lines, surplus lines are removed; leftover values are appended as new lines after the host's last existing line (with one extra blank `ConfigLine(line="")` separator when the host block ends at the file end); key casing is remapped to canonical ssh parameter casing where recognized.
   - `unset(host, *args)` — delete all lines for the host whose lowercase key is in the given arguments.
   - `rename(old, new)` — ValueError if new host already exists; updates the `Host` line text, line host assignment and the host cache.
   - `add(host, **kwargs)` — ValueError if host exists; appends a blank line, `Host <host>` line, then one line per value per kwarg, then a trailing blank line.
   - `remove(host)` — ValueError if host not found; removes the host's line range including comment lines inside that range.
   - `config(filter_includes=False)` → the file text reconstructed from stored lines, joined with `\n`; when filtering, lines whose lowercase key is `include` are omitted.
   - `write(path)` — write `config()` to the file.
   - Setting or unsetting a nonexistent host, or a parameter named `host` (case-insensitive), raises `ValueError`.

7. Provide `read_ssh_config(master_path)`: read the master file, then process any `Include` directive values by globbing against the master file's directory (after `os.path.expanduser` on the include pattern), reading each resolved file (deduplicate by processing a work queue so each file is read once, BFS-style), and return an aggregate multi-file configuration object that:
   - `hosts()` returns the union of hosts across files;
   - `host(host)`, `set`, `unset`, `remove` operate on the first file containing the host (`set`/`unset`/`remove` of a missing host raise `ValueError`, `host` returns `{}`);
   - `rename` raises if the new name exists anywhere, and renames in every file where the old host appears;
   - `add(host, **kwargs)` appends to the master (first) file;
   - `config()` concatenates each file's `config(filter_includes=True)`, i.e. one merged file without Include lines;
   - `write(path)` writes that merged config;
   - `save()` writes each parsed file back to its original path.

8. Deliver packaging metadata: the project describes itself with `pyproject.toml` using the `flit_core >=3.2,<4` build backend (installed as `flit_core==3.12.0`), name `sshconf`, dynamic version and description, `requires-python = ">=3.5"`, no runtime dependencies, MIT license classifier, `ssh`/`config` keywords, README as long description, and a test extra with `pytest`, `pytest-cov`, `codecov`. Since the module docstring is used by the build backend, keep an accurate one-lin