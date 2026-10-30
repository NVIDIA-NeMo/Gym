# Description

OfficeEnv simulates a set of connected SaaS apps (Slack, Gmail, Google Calendar, Google Sheets,
HubSpot, Jira and about a hundred more; `GET /officeenv/apps` lists them) for each rollout. A
row's `officeenv` task spec says which apps are connected, the data they hold, the acting user,
how the apps reach the policy (`tools`, `api_search`, `cli`, `mcp`), the faults and data
injections that happen during the episode, and the final-state checks that produce the reward.
Episodes are deterministic: the same spec and the same tool calls give the same world and reward.

The environment is the `officeenv` package (pydantic and the standard library), a separate
repository. This directory is the Gym integration: `app.py` routes Gym's endpoints to OfficeEnv's
framework-neutral episode service (`officeenv.service.EpisodeService`), and `requirements.txt`
pins the release it runs.

## The OfficeEnv version

`requirements.txt` pins a release tag:

```
officeenv @ git+ssh://git@gitlab-master.nvidia.com:12051/dmolinari/officeenv.git@v1.9.0
```

OfficeEnv is not published yet, so building this server's venv needs SSH read access to that
repository (`git+https://gitlab-master.nvidia.com/...` works instead where a git credential helper
or `~/.netrc` holds a token). On a cluster, build the venv on a login node and reuse it
(`skip_venv_if_present=true`), or bake it into the container with `gym env prefetch`.

Release numbers follow OfficeEnv's `docs/RELEASING.md`: a patch release changes no verdict for an
existing row, a minor one may add features or move verdicts (each listed in OfficeEnv's
`CHANGELOG.md`), a major one may refuse rows emitted by an older major version.

To move to a new release:

1. change the tag in `requirements.txt`;
2. emit `data/example.jsonl` again with that release (below), collect `data/example_rollouts.jsonl`
   again (see Run), and regenerate `data/example_metrics.json`:
   `gym dataset collate "+config_paths=[resources_servers/officeenv/configs/officeenv.yaml]" +output_dirpath=data/officeenv +mode=example_validation`;
3. `gym env test +entrypoint=resources_servers/officeenv +should_validate_data=true`.

To run an unreleased OfficeEnv, put its source first on `PYTHONPATH` when starting the servers
(`PYTHONPATH=<officeenv checkout>/src gym env start ...`). Gym keeps the caller's `PYTHONPATH` for
every server, so that tree shadows the pinned release, and `/verify` reports its version.

## Rows

```json
{"responses_create_params": {"input": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "tools": [...]},
 "officeenv": {"task_id": "...", "seed": 1, "actor": {...}, "apps": {...}, "exposure": {"mode": "api_search"}, "checks": [...]},
 "officeenv_version": "1.9.0",
 "task_id": "..."}
```

`officeenv emit <task.json>... -o rows.jsonl` writes them (`officeenv.emit.gym_row` from Python).
The tool list is the one the emitting version serves for the spec, and `officeenv_version` records
that version:

- `/seed_session` refuses a row from another major version (400, `incompatible_row`); rows
  without a stamp (emitted before 1.0.0) run as they are;
- a row whose tool list differs from what the server serves (a renamed tool, other parameters)
  still runs; `/seed_session` and `/verify` report the tools in `tool_drift`.

The example rows are five of OfficeEnv's built-in examples: one per exposure mode, plus a task
with a scripted user.

```bash
E=<officeenv checkout at the pinned tag>/src/officeenv/examples
officeenv emit $E/renewal_escalation.json $E/schedule_kickoff.json $E/vendor_terms_review.json \
  $E/ticket_triage.json $E/sprint_handover.json -o data/example.jsonl
```

The example rollouts were collected on 2026-10-28 with `azure/openai/gpt-5.5` against OfficeEnv
1.9.0, through this server and the simple agent (concurrency two). All five score 1.0. Every
error the metrics count is an authored fault (a transient error, rate limits) or a tool mistake
the model recovered from. They are integration smokes, not a benchmark: `schedule_kickoff` scores
0.75 whenever the model books its slot over a meeting injected during the episode
(`gcalendar.no_conflicts`), as it did on earlier runs.

Training and validation sets are not committed.

## Artifacts and corpora

A row may name its TaskSpec's large strings (a seeded workbook's base64, a long CSV) by SHA-256
instead of carrying them, in OfficeEnv's artifact format (`officeenv.artifacts`):

```json
{"$artifact": "sha256:<64 hex digits>", "encoding": "base64", "bytes": 10240}
```

A corpus is one directory, its rows files beside the store of the artifacts they name:

```
<corpus>/<rows>.jsonl
<corpus>/artifacts/<64 hex digits>
```

so it moves as one directory and is read with only its rows path changed.
`python -m officeenv.artifacts externalize rows.jsonl -o <corpus>/rows.jsonl` writes one, and
`check`, `pack` and `unpack` verify a corpus, write it as one archive and install it.

The server reads the references back at seed, for an Agent's seed and an Environment Server's
alike, from the stores in `artifacts_dirs`: absolute paths (the server runs in its own
directory), searched in order. Each artifact is checked
against its hash and size before OfficeEnv reads the TaskSpec. A row whose artifacts cannot be
read is refused with 400 (`missing_artifact`, `artifact_mismatch` or `invalid_artifact`) and
nothing is seeded; a row without references seeds as before. To serve a corpus:

```bash
gym env start --resources-server officeenv --model-type openai_model \
  "++officeenv_resources_server.resources_servers.officeenv.artifacts_dirs=[/data/corpus/artifacts]"
```

## Run

```bash
gym env start --resources-server officeenv --model-type openai_model
gym eval run --no-serve --agent officeenv_simple_agent \
  --input resources_servers/officeenv/data/example.jsonl \
  --output results/officeenv_rollouts.jsonl
```

The simple agent stops after `max_steps` (50) turns, or once the model answers without calling a
tool. Tasks explored through `api_search` spend a search and a fetch on every endpoint before
calling it, so long ones need more:
`+officeenv_simple_agent.responses_api_agents.simple_agent.max_steps=150`.

## Environment Server episodes and MCP

An Environment Server (for example `single_agent_turn`) seeds with a `ResourcesSeedSessionRequest`
whose `task_data` holds the row's fields other than `responses_create_params`. The OfficeEnv
session takes the request's `resources_session_id`, which the seed's session cookie also names;
the agent calls the same `POST /<tool_name>` routes with that cookie, and `/verify` grades the
episode as for an Agent. Seeding the same ID again for the same episode returns the same session,
and `/close_session` frees a world that verification has not. After a close, a seed of the same
ID is refused for `session_close_retry_window_seconds`. These seeds carry no tool list, so they
report no `tool_drift`.

With `expose_tools_over_mcp: true`, Gym's MCP endpoint serves every tool of OfficeEnv's `tools`
exposure under its tool name, and each episode's MCP token (in the seed response) admits only
that episode's tools. MCP clients read the tool definitions from that endpoint rather than from
the row, so these episodes need rows in the `tools` exposure with the default tool-name
separator; a seed whose tools differ from what the endpoint serves fails with 400
(`unsupported_exposure`).

## Sandboxed harnesses

Sandbox support is opt-in: without `sandbox_provider` the server runs as above.
`configs/officeenv_sandbox_episodes.yaml` is the resources server for sandboxed episodes, and each
harness config composes it with a harness in agent-session mode behind the
`single_agent_turn_legacy` Environment Server: `configs/officeenv_sandbox.yaml` runs Pi, with the
tools over MCP as above; `configs/officeenv_sandbox_codex.yaml` (Codex CLI),
`configs/officeenv_sandbox_miniswe.yaml` (mini-SWE), `configs/officeenv_sandbox_opencode.yaml`
(OpenCode), `configs/officeenv_sandbox_openclaw.yaml` (OpenClaw) and
`configs/officeenv_sandbox_terminus_2.yaml` (Terminus 2) run rows whose apps are commands (below),
with no MCP tools. Each episode also:

1. borrows a sandbox: the seed starts one from `sandbox_config.image` (built from
   `sandbox/Dockerfile`), or from another configured image when the row needs tools that one
   lacks (see Images and what a row needs), with Gym's sandbox provider named by
   `sandbox_provider`, and lends it to the harness as `SandboxAccess`. Seeding the episode again
   lends the same sandbox;
   verification stops it (or `/close_session`, when verification did not), and so does shutting
   the server down;
2. moves file content between the workspace and the file apps (Google Drive, Box, SharePoint,
   OneDrive) without passing it through the model: the sandbox gets `COMPANY_API_URL` and
   `COMPANY_API_TOKEN`, in its environment and in `/etc/company-api.env` for a harness whose
   shell does not inherit the sandbox's environment (Codex starts its shell with `PATH`, `HOME`,
   the locale and the XDG folders alone; the apps' commands read the file, and the guide says
   `. /etc/company-api.env`), and `README.md` in the workspace documents the file API for the
   episode's file apps (`GET`/`PUT /v1/files/{app}/{file_id}`,
   `POST /v1/files?app=&name=&parent=&mime_type=`). Files and folders are named by id: the server
   reads what an app's op also wants (SharePoint's library, folder and name) from the world.
   SharePoint uploads need a folder (`parent`); `mime_type` is Drive's. Text and binary files (an
   XLSX workbook, say) move byte for byte; uploads are limited to 20 MB; an app's error comes back
   under its own HTTP status (a missing file is 404, a name Box already holds 409);
3. keeps their content out of the conversation on the apps' own tools too
   (`sandbox_config.inline_file_bytes`, 8192 by default):
   - a download (Drive's `download_file`, `export_file` and `get_revision`, Box's
     `get_file_content`, SharePoint's `get_item_content`, OneDrive's `get_content` and
     `get_version_content`) saves the file in the workspace at `<folder>/<file id>/<name>`
     (`drive/`, `box/`, `sharepoint/`, `onedrive/`; a Drive revision under
     `revisions/<version>/`, a Box or OneDrive version under `versions/<version>/`, an export with
     its type's extension), mode 644, and answers with `savedTo` and `savedBytes`. The answer
     carries the content as well only for a text file of at most `inline_file_bytes` bytes; a
     longer text gets its first 2,000 characters as `preview`, a binary file nothing but its path;
   - the uploads (Drive's `create_file` and `update_file`, Box's `upload_file` and
     `upload_file_version`, SharePoint's `upload_content`, OneDrive's `upload_file`) refuse binary
     content, and text over `inline_file_bytes` bytes, pasted into the call
     (`upload_from_workspace`), with the file API command that sends the file instead;
   - Gmail's attachments move through the workspace too, since Gmail has no file API:
     `get_attachment` saves one at `gmail/<message id>/<attachment id>/<name>` and answers as the
     downloads above, and an attachment of `send_message`, `create_draft` or `update_draft` may
     name a workspace file by its path (`{"path": "/workspace/totals.xlsx", "filename"?,
     "mimeType"?}`; a relative path is in the workspace), or as a bare string that writes a path
     or names a file the workspace holds. The server reads that file from the sandbox before the
     call runs and the call carries its bytes. Each path is checked in the sandbox first (`stat`,
     which follows no final link): only a regular file of at most 20 MB is downloaded, nothing
     when a message names more than 20 files or more than 25 MB together, and the copy is checked
     again, since `docker cp` copies a link as a link, which the server would read on its own
     host. A download that fails attaches nothing. Pasted binary content, and text over `inline_file_bytes` bytes, are
     refused (`upload_from_workspace`), and so is an attachment that names a file without its
     content (`attach_from_workspace`), each with the attachment that names the workspace file;
   - these tools are described as they behave here, in place of OfficeEnv's words
     (`contentBase64`, `content_base64`), and so are the uploads' `content` and
     `content_base64` parameters and Gmail's `attachments`; `README.md` says the same.
     `/verify` reports `sandbox_files_saved`, `sandbox_bytes_saved`,
     `sandbox_uploads_refused`, `sandbox_files_attached` and `sandbox_bytes_attached`, over all
     five apps.

   The file API, short text, calls without content and episodes without a sandbox are unchanged;
   `inline_file_bytes: null` restores full answers. The server wraps the episode's OfficeEnv
   session for this (`ContentInWorkspace`), so OfficeEnv itself is unchanged. A model that pulls a
   workbook through the tool reads its bytes as base64 on every later request, and one that copies
   file content by hand gets it wrong: in earlier pilots, models did both;
4. ends when Pi's run ends: `finish_tool_names: [user_complete_task]`, its timeout, or the model
   stopping. `/verify` first closes the episode's tools and file API and stops its sandbox, then
   grades the world as for any row: a process the harness left running in the background can act
   until verification starts, but not while the world is graded. A failed stop is logged and
   grading goes on; `/close_session` stops the sandbox then.

### Images and what a row needs

A row may say what its task needs in the sandbox beyond the shell, curl and Python 3.12 with
pandas and openpyxl, as `sandbox_needs` (`TOOL_SETS`): `documents` is python-docx, python-pptx,
pypdf, pdfplumber and reportlab, and Poppler's `pdftotext`, `pdfinfo` and `pdftoppm`; `office` is
LibreOffice without a display, which converts documents (`soffice --headless --convert-to pdf`,
or `docx`, `xlsx`, `pptx`) and recalculates a workbook's formulas as it converts it.
`sandbox_config.provides` says what `sandbox_config.image` has, and `sandbox_config.images` lists
more images with what each provides. A row's episode gets the first image that provides all it
needs, `image` first, and a row that no configured image serves is refused at seed, before its
world is built (400 `unsupported_needs`). The workspace's `README.md` lists what the episode's
image has, and with `block_internet` says that nothing else can be installed.
`sandbox/prepare_rows.py` copies an example task's `sandbox_needs` to its row.

`sandbox/Dockerfile.documents` adds both to an episode image, with pinned Python packages and
LibreOffice 7.4 and Poppler 22.12 from Debian 12. It builds on `sandbox/Dockerfile`'s image by
default, or on another with `--build-arg BASE=<image>`, such as a training image with a harness
staged in it:

```bash
docker build -t officeenv-sandbox:pilot resources_servers/officeenv/sandbox
docker build -f resources_servers/officeenv/sandbox/Dockerfile.documents \
  -t officeenv-sandbox:documents resources_servers/officeenv/sandbox
```

Serve it to every row:

```yaml
sandbox_config:
  image: officeenv-sandbox:documents
  provides: [documents, office]
```

or only to the rows that need it, since it is about 700 MB larger:

```yaml
sandbox_config:
  image: officeenv-sandbox:pilot
  images:
    - image: officeenv-sandbox:documents
      provides: [documents, office]
```

A cell whose sandboxes have no internet pulls the image from a registry, pinned by digest: its
tools cannot be installed at sandbox start, as the OpenSandbox config installs pandas and openpyxl.

### Images per harness

A harness that runs in the sandbox installs its runtime there at session start (Pi and Codex
fetch Node and their package), which a sandbox without the internet cannot do.
`sandbox/Dockerfile.harness` builds one image per harness on the documents image: each target
runs the harness adapter's own installer as the adapter runs it (`sandbox/harness_runtime.sh`),
so the runtime is where the adapter looks for it at session start and the adapter downloads
nothing. Build from the Gym checkout's root (`Dockerfile.harness.dockerignore` sends only the
installers and that script):

```bash
docker build -f resources_servers/officeenv/sandbox/Dockerfile.harness --target pi \
  -t officeenv-sandbox:documents-pi .
```

The targets are `pi`, `codex`, `openclaw`, `opencode` and `terminus-2` (Terminus 2 runs its agent
on Gym's host and needs only `tmux` in the sandbox). The versions are the adapters' configured
ones (`PI_VERSION`, `CODEX_VERSION`, `OPENCLAW_VERSION` and `OPENCODE_VERSION` override them), and
the runtime folders follow the adapters at this Gym revision: rebuild when an adapter changes.
Check a built image without a network, where the adapter's installer must find the runtime and
the harness must start:

```bash
docker run --rm --network none -v "$PWD:/gym:ro" officeenv-sandbox:documents-pi \
  bash /gym/resources_servers/officeenv/sandbox/harness_runtime.sh pi /gym/responses_api_agents
```

A run names its harness's image in `sandbox_config.image`, with `provides: [documents, office]`.
An Environment Server binds one harness to one resources server, so in a mix of harnesses each
harness gets its own OfficeEnv server with its own image. Hermes, Stirrup and Claude Code also
reuse a runtime they find in the sandbox, but name its folder after their agent server's own
install (the Hermes commit, Stirrup's version, Claude Code's binary digest), so they have no target
yet; mini-SWE and Kilo Code install at every session start, whatever the image holds. Pi, Codex,
OpenClaw and OpenCode keep their runtime under `/tmp`, which a provider that mounts a fresh `/tmp`
would hide. An image keeps only the runtime: once the installer and the harness have run, the
script removes what they leave behind, the archives the install unpacked, npm's cache and logs
(in `~/.npm`, or in the runtime folder for an installer that moves `HOME`, as Codex's and
OpenClaw's do), Node's compile cache in `/tmp` and the helper links Codex makes in `~/.codex`.

### Tools, commands or endpoints

The row's exposure decides how the harness reaches the apps (`SANDBOX_SURFACES`); the world,
the checks and the reward are the same for all three:

- `tools`: one MCP tool per op, as above; the episode's MCP token admits its row's tools;
- `cli`: one command per app in the sandbox, `<app> <command> [--parameter=value ...]`, with
  `<app> --help` and `<app> <command> --help`, read as OfficeEnv's CLI exposure reads a command
  line. The seed writes `sandbox/app_command.sh` (POSIX sh and curl, without its comments) to
  `/usr/local/lib/company-cli/app` and links it as `/usr/local/bin/<app>` for each connected app.
  A command sends its arguments to `POST /v1/cli` with the episode's token and prints the result
  on stdout (exit 0) or the app's error on stderr (exit 1). The MCP token admits no tool, so a
  harness with nothing but a shell runs these rows (`resources_tool_transports: []`). A call to
  `/v1/cli` that does not come from an app's command (its `<app>-cli/1.0` User-Agent) is refused
  with a pointer to the command, and counted;
- `api_search`: the MCP tools `api_search` and `api_fetch` find and document endpoints, with each
  endpoint's HTTP path and the curl command that calls it, and the harness calls
  `POST /v1/<app>/<op>` with the op's parameters as a JSON object. An endpoint answers with the
  app's JSON, or with its error under the error's HTTP status.

The file API and the content kept out of the conversation (downloads saved, pasted uploads
refused, Gmail's attachments by path) serve every surface: a command's `--help` and `api_fetch`'s
documentation describe the ops that move content as they behave here, as the tools' descriptions
do. The workspace's `README.md` is written for the episode's surface, and
`sandbox_config.mcp_tool_prefix` names the MCP tools in it as the harness lists them (`company_`
for Pi). `/verify` adds `sandbox_commands`, `sandbox_commands_refused` and
`sandbox_endpoint_calls`.

Rows come from OfficeEnv's sandbox example tasks: `sandbox/prepare_rows.py` writes
`data/sandbox_pilot.jsonl` with the OfficeEnv it imports (its five sandbox tasks). A
row holds a system and a user message and the TaskSpec, and no tool list, since the harness takes
its tools from the MCP grant. `--exposure cli` or `--exposure api_search` writes the same tasks in
that exposure, with a prompt that says how the apps are reached, to
`data/sandbox_pilot_<exposure>_prepare.jsonl`; `--tool-prefix` names the MCP tools as the harness
lists them (`company_`, Pi's, by default; `mcp__company__` for Codex).

To run the commands' rows with another harness, use its config and pass `--agent` to the
collection, for Codex CLI:

```bash
python resources_servers/officeenv/sandbox/prepare_rows.py --exposure cli
gym env start --config resources_servers/officeenv/configs/officeenv_sandbox_codex.yaml \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml <model and host arguments as above>
gym eval run --no-serve <the same --config and arguments> --agent codex_agent \
  --input resources_servers/officeenv/data/sandbox_pilot_cli_prepare.jsonl --output results/codex_cli.jsonl
```

mini-SWE, OpenCode, OpenClaw and Terminus 2 run the same way with their configs and
`--agent miniswe_sandboxed_agent`, `opencode_agent`, `openclaw_agent` or
`terminus_2_sandboxed_agent`. Pi runs these rows with `configs/officeenv_sandbox.yaml` and
`++single_agent_turn_legacy.environment_servers.single_agent_turn_legacy.resources_tool_transports=[]`,
and `api_search` rows with its MCP transport as it is; on both, no MCP tool ends the task, so Pi
takes `++pi_agent.responses_api_agents.pi_agent.finish_tool_names=[]` and ends when the model stops.
Any of Gym's harnesses that borrow an episode's sandbox and run a shell can be paired the same way
(the agent's config, `agent_server.name`, no MCP transports); what each needs in the sandbox image
(its own runtime, when sandboxes have no internet) is the harness's.

What a sandbox reads of this server names a company's API, never the environment behind it or
Gym: the variables above, the `/v1/...` routes, the workspace's `README.md`, the answers and
errors of every route a sandbox reaches, and the session cookie that answers set (`sid_<hash>`,
`get_session_cookie_name`). The config's `mcp_server_name: company` and `mcp_bearer_token: true`
make the MCP endpoint an ordinary one too: Pi lists it as `company` and names its tools
`company_<tool>` (at most 64 characters; the longest tool name is 56), the session token goes as
a standard bearer token, and `tools/list` without it lists nothing. `sandbox/prepare_rows.py`'s
system prompt names the tools the same way. On OpenSandbox, whose exec daemon keeps each command's
output in the sandbox, `configs/officeenv_opensandbox.yaml` has the provider's post-create probe
print `ready` (`sandbox.opensandbox.probe`) instead of its default marker, which names Gym. What a
sandbox still reads of Gym comes from outside this server: the policy's URL and Pi's own files and
environment (see Known limits).

A sandbox can reach this server, so sandbox mode guards it:

- seeding, closing, verification, `GET /officeenv/apps` and the tool routes answer only callers
  from this node and from `trusted_networks` (CIDR, empty by default), and others as routes that
  do not exist (404); no generated docs (`/openapi.json`, `/docs`) are served. Sandboxes keep the
  MCP endpoint, the file API and their surface's routes (`/v1/cli`, `/v1/<app>/<op>`), whose
  tokens name their episode. When Gym's other servers call this one from other nodes, list their
  network in `trusted_networks`;
- session cookies and MCP tokens are signed with a secret drawn when the server starts
  (`get_session_secret`), not with Gym's default key, which anyone can derive from the server's
  name, so code in a sandbox cannot sign a token for another episode or for more tools. The secret
  lives in one process, so sandbox mode runs one worker (`num_workers: 1`).

The sandbox must reach this server and the model server. On a workstation the servers bind
loopback, which a container cannot reach (and a host firewall may drop connections to the host's
other addresses): set `sandbox_config.host_relay` to a host alias that reaches the host's
loopback (colima: `host.lima.internal`; Docker Desktop: `host.docker.internal`), and each sandbox
runs `sandbox/relay.py`, which forwards this server's and the model servers' ports. Relayed
connections arrive from the host's loopback, so the caller check above admits them; the signed
tokens still apply. Use the relay on a workstation only.

```bash
docker build -t officeenv-sandbox:pilot resources_servers/officeenv/sandbox
gym env start --config resources_servers/officeenv/configs/officeenv_sandbox.yaml \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
  --model-type inference_provider --model <model> --model-url <url> '+policy_api_key=...' \
  ++oe.resources_servers.officeenv.sandbox_config.host_relay=host.lima.internal
gym eval run --no-serve <the same --config, --model and ++ arguments> --agent pi_agent \
  --input resources_servers/officeenv/data/sandbox_pilot.jsonl --output results/sandbox_rollouts.jsonl
```

On a cluster, `configs/officeenv_opensandbox.yaml` composes the same episodes with the OpenSandbox
provider, as Gym's SWE-bench and Terminal-Bench runs use it: the servers run on a cluster node,
the sandboxes on the OpenSandbox cell, and `use_absolute_ip` with
`absolute_ip_server_types: [responses_api_models, resources_servers]` advertises the node's address
for the two servers a sandbox calls (the head server stays on loopback; no relay). Credentials go
in `env.yaml` under `sandbox.opensandbox.connection`. Until `sandbox/Dockerfile`'s image is
published to a registry the cell pulls from, the config uses a stock `python:3.12-bookworm` image
(pinned by digest) and installs pandas and openpyxl in the provider's setup commands.

`sandbox_config.block_internet: true` takes the internet away from each sandbox: the server sets
OpenSandbox's egress policy (`provider_options.network_policy`, built with Gym's
`restricted_network_policy`) to refuse every outbound connection except to this server (the MCP
endpoint and the file API) and the model servers, whose addresses it reads from Gym's config; a
model server on loopback is left out. It fails closed: another sandbox provider, a server address
a sandbox cannot reach, `host_relay`, or a `network_policy` of your own are refused. On an
OpenSandbox cell the policy is enforced on addresses, not only on names: in a probe on 2026-10-27,
public names did not resolve, bare public IPs and DNS over HTTPS timed out, and the node still
answered. It is off in both configs, because their sandboxes download at start (Pi's Node runtime
and package, and the OpenSandbox config's pandas and openpyxl); turning it on needs an image with
those installed. Gym's sandboxed agents restrict only sandboxes they start themselves, so for these
episodes, whose sandbox the resources server owns, the server sets the policy.

Before the first run on a cell, `sandbox/probe_opensandbox.py` checks it from the node that will
host Gym's servers: it starts one sandbox as the config would and reports the setup time, the task
tools, a call back to the node with the source address the node sees (keep that address out of
`trusted_networks`), internet egress and the time Pi's install takes. Its docstring has the
command.

Known limits: Pi installs Node and its own package in every sandbox at session start (network
access to nodejs.org and npm); token-ID capture through Pi needs a vLLM model server and is not
validated here. Pi, Codex and Hermes take MCP tool grants in sandbox sessions; OpenCode and Claude
Code refuse them, so they run only the commands' rows. Codex installs Node and its package, and
mini-SWE its Python packages, in each sandbox at session start. What a sandbox still reads of Gym:

- the policy's URL, which Gym's agents hand the harness with Gym's own path names:
  `/ng-rollout/<id>` from rollout correlation (`ROLLOUT_PATH_PREFIX`), `/training-token-capture`
  with token capture (`TOKEN_CAPTURE_PATH_SEGMENT`), and `/ng-trace/<traceparent>` while traces
  export. Pi writes it into its `models.json` in the sandbox, where a model can read it (it is not
  in Pi's environment or process arguments). The test of what a sandbox reads lists it among its
  findings; making the two path names configurable is a further core change;
- Pi's own names: its session and runtime directories, the `NEMO_GYM_PI_*` variables its shell
  inherits, the model provider `nemo`, its supervisor's process names, and a system prompt that
  states the sandbox clock's date. Changing those is Pi agent work;
- the other harnesses' own names, which their Gym adapters fix and the harness images keep where
  the adapters look: the session and runtime folders of Codex (`/tmp/nemo-gym-codex-sessions/<id>`,
  `/tmp/nemo-gym-codex-node-22.19.0-<version>`), OpenClaw (`/tmp/nemo-gym-openclaw-sessions/<id>`,
  `/tmp/nemo-gym-openclaw-node-22.19.0-<version>`) and OpenCode
  (`/tmp/nemo-gym-opencode-sessions/<id>`, `/tmp/nemo-gym-opencode-runtime-<version>` and its
  `.lock`), and mini-SWE's task file and working folder (`/tmp/.nemo-gym/task.json`,
  `/tmp/nemo-gym-miniswe-<id>`). The adapters take no other folder, so changing those is adapter
  work;
- the system prompts the harnesses send with every model call, which name those folders: Pi's
  points to its documentation (three paths in its runtime folder), Codex's to its skills (five
  paths in its session folder), OpenClaw's to its documentation, and OpenCode's to its
  instructions file, and it gives the model's ID as `nemo_gym/dummy_model` (the provider and
  model names OpenCode's adapter configures). Terminus 2's names none. Changing those is adapter
  work as well.

The caller check is not a network boundary: a gateway that exposes only MCP and the file API to
sandboxes would be one.

## Verification

`/verify` grades the final state of the rollout's session and frees it. Besides `reward` (a
guarded weighted mean by default), it returns:

- `officeenv_checks`: per check, pass or fail with observed and expected values;
- `officeenv_metrics`, with flat numeric copies (`checks_passed`, `checks_total`,
  `guards_failed`, `policy_calls`, `errors`, `faults`, `injections`, `discovery_calls`,
  `unknown_tool_calls`, `write_events`, `tool_drift`) for metric aggregation;
- `officeenv_scenario_log`, `officeenv_integration_log`, `officeenv_timer_failures`: which faults
  and injections fired (or were held), cross-app effects, and timers that failed;
- `officeenv_server_version`: the OfficeEnv version that graded the rollout.

Checks on the policy's replies read every message it sent, including any it sent after completing
the task. A rollout without a session to grade (never seeded, verified already, closed, or lost
to a server restart) gets reward 0 with `mask_sample: true` and `failure_kind: session_lost`, and
a tool call without a session answers 404 with the same `failure_kind`.

## Training length penalty

`length_penalty.enabled=true` on the resources server deducts AutomationBench's training length
penalty (`benchmarks/automationbench`, same knobs and defaults) from rewards above a threshold:

```python
if reward <= reward_threshold:  # 0.5
    return reward
return reward - min(max_penalty, turns / turn_scale + tokens / token_scale)  # 0.5, 200, 200000
```

Turns are the model's responses in the episode; tokens are the completion tokens it generated
over all of them, reasoning included, from the response usage the simple agent accumulates (the
penalty fails explicitly when a rewarded episode has none). With the penalty on, `/verify` also
returns

- `reward_components`: `{"task_reward": <OfficeEnv's reward>, "length_penalty": -<deduction>}`,
  which sum to `reward`, so a trainer can apply the penalty on its own schedule (NeMo RL:
  `grpo.phasic_reward_components` with `components=[length_penalty]`);
- `task_reward`, `length_penalty` (the deduction), `model_turns` and `generated_tokens` as
  metrics.

Set it on training jobs only; with it off, rewards are OfficeEnv's. From NeMo RL, at
AutomationBench's half strength:

```
++env.nemo_gym.officeenv_resources_server.resources_servers.officeenv.length_penalty.enabled=true
++env.nemo_gym.officeenv_resources_server.resources_servers.officeenv.length_penalty.turn_scale=400
++env.nemo_gym.officeenv_resources_server.resources_servers.officeenv.length_penalty.token_scale=400000
```

# Licensing information
Code: this server, Apache 2.0; the `officeenv` package, NVIDIA, not yet published (its open-source release is under review)
Data: the example rows are synthetic, emitted from OfficeEnv's built-in examples

Dependencies
- nemo_gym: Apache 2.0
- officeenv: NVIDIA, not yet published; its only dependency is pydantic (MIT)
