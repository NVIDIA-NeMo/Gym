# Description

OfficeEnv's episodes for sandboxed harnesses. `app.py`'s `OfficeEnvSandboxResourcesServer` subclasses the OfficeEnv
resources server (`resources_servers/officeenv`) and keeps its rows, routes, verification and options, which
[its README](../officeenv/README.md) describes. Each episode session here also borrows a sandbox, where a coding-agent
harness works on the task: the sections below cover what that adds. An agent-run seed (the simple agent's `/run`) gets
no sandbox and runs as on the OfficeEnv server.

## The OfficeEnv version

This server runs the OfficeEnv release the OfficeEnv server's README names ("The OfficeEnv version"), which installs
apart from both servers' locked dependencies, the same way, into this server's venv:

```bash
uv pip install --python resources_servers/officeenv_sandbox/.venv/bin/python "officeenv @ git+${OFFICEENV_GIT_URL}@v1.14.0"
```

Workspace mode and the file call (`officeenv.files`) need OfficeEnv 1.13.0 or later: with an older release, or none,
the server stops at start naming the step that installs it.

## Example data

`data/example.jsonl` holds the first five of `data/sandbox_pilot.jsonl`'s rows, and
`data/example_rollouts.jsonl` Pi's rollouts of them on Docker sandboxes (`configs/officeenv_sandbox.yaml`),
collected on 2026-11-13 with `azure/openai/gpt-5.5` against OfficeEnv 1.14.0. All five score 1.0; the
episodes saved 8 files to their workspaces, uploaded 5 and attached 1, and no sandbox was left running.

## Sandboxed harnesses

`configs/officeenv_sandbox_episodes.yaml` is the resources server for sandboxed episodes, and each
harness config composes it with a harness in agent-session mode behind the
`single_agent_turn_legacy` environment server: `configs/officeenv_sandbox.yaml` runs Pi, with the
tools over MCP as on the OfficeEnv server; `configs/officeenv_sandbox_codex.yaml` (Codex CLI),
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
   the server down. The sandbox's TTL runs from the seed, so where the harness caps the episodes
   it runs at once (`concurrency`: Pi, Codex, OpenCode, OpenClaw), its config has the Environment
   Server admit no more (`max_concurrent_episodes` follows it), and the rest wait there, before
   their seed, for up to a day (`queue_timeout_seconds`). mini-SWE and Terminus 2 run every
   episode they are given;
2. moves file content between the workspace and the file apps (those of the episode's apps that
   OfficeEnv gives a file API: Google Drive, Box, SharePoint, OneDrive) without passing it through
   the model: the sandbox gets `COMPANY_API_URL` and `COMPANY_API_TOKEN`, in its environment and in
   `/etc/company-api.env` for a harness whose shell does not inherit the sandbox's environment
   (Codex starts its shell with `PATH`, `HOME`, the locale and the XDG folders alone; the apps'
   commands read the file, and the guide says `. /etc/company-api.env`), and `README.md` in the
   workspace documents the file API for the episode's file apps (`GET`/`PUT
   /v1/files/{app}/{file_id}`, `POST /v1/files?app=&name=&parent=&mime_type=`), with OfficeEnv's
   note on each. The routes are OfficeEnv's file call (`EpisodeService.file_download`,
   `file_create`, `file_replace`), the agent's own action, which maps files and folders named by
   id onto each app's ops. Text and binary files (an XLSX workbook, say) move byte for byte;
   uploads are limited to 20 MB; an app's error comes back under its own HTTP status and body (a
   missing file is 404, a name Box already holds 409, an app the task makes unavailable 503);
3. keeps their content out of the conversation on the apps' own calls too: every sandboxed
   episode runs in OfficeEnv's workspace mode from its seed (`officeenv.files`,
   `EpisodeService.enable_workspace`), with the workspace's folder, an inline limit
   (`sandbox_config.inline_file_bytes`, 8192 by default; `null` takes text of up to 20 MB inline,
   while other files still move through the workspace) and the file API's commands for a refused
   upload. OfficeEnv decides, for every app that declares where its content crosses its API, what
   the mode does, and describes it in every exposure's own words:
   - a download is saved in the workspace and answered with `savedTo` and `savedBytes`, the
     content as well only for a text file of at most the inline limit (a longer text gets its
     start as `preview`); the server writes each saved file into the sandbox before it answers
     (`Reply.workspace_writes`), mode 644;
   - an upload takes a workspace file by its path (`content_path`), and an attachment by its
     `path`; pasted binary content, and text over the limit, are refused
     (`upload_from_workspace`), naming the path form and the file API command for the call's
     target. The server reads each file a call names from the sandbox before the call runs
     (`EpisodeService.workspace_reads`). Each path is checked in the sandbox first (`stat`, which
     follows no final link): only a regular file of at most 20 MB is downloaded, nothing when a
     call names more than 20 files or more than 25 MB together (or more than the op's own caps,
     when stricter), and the copy is checked again, since `docker cp` copies a link as a link,
     which the server would read on its own host. A download that fails reads nothing;
   - `/verify` reports `sandbox_files_saved` and `sandbox_bytes_saved` (the writes that landed),
     and `sandbox_uploads_refused`, `sandbox_files_attached`, `sandbox_bytes_attached`,
     `sandbox_files_uploaded` and `sandbox_bytes_uploaded` (OfficeEnv's
     `EpisodeService.workspace_stats`), and passes on OfficeEnv's trace of the mode
     (`officeenv_workspace`); with `workspace_trace_store` set, the bytes the server hands an
     episode are kept there, so the trace replays without a sandbox. A write the store fails
     (a full disk) is logged and counted in `sandbox_trace_failures`, and never fails the call.

   Episodes without a sandbox are unchanged. A model that pulls a workbook through the tool reads
   its bytes as base64 on every later request, and one that copies file content by hand gets it
   wrong: in earlier pilots, models did both;
4. ends when Pi's run ends: `finish_tool_names: [user_complete_task]`, its timeout, or the model
   stopping. `/verify` first closes the episode's tools and file API and stops its sandbox, then
   grades the world as for any row: a process the harness left running in the background can act
   until verification starts, but not while the world is graded. A failed stop is logged and
   grading goes on; `/close_session` stops the sandbox then.

The server lends its sandboxes through Gym's `LentSandboxes`: the harness gets an `operate`
descriptor, which lets it use the sandbox but not destroy it, and only this server stops it. When
the server shuts down (a job's end, Ctrl-C), it stops every sandbox at once, those of failed seeds
whose stop failed among them, within Gym's `server_shutdown_timeout_seconds` less 2 s for the
exit; no seed starts a sandbox from then on (503). The sandboxed configs set that timeout to 30 s,
since Gym's default of 1 s cannot stop a collection's sandboxes; it is run-wide, so every server
of the run gets it. Any sandbox that may still run is logged at ERROR with its session, for a
reaper. Gym cancels the requests in flight first, and a request whose client gives up: a seed
cancelled so still finishes starting its sandbox (within 30 minutes), then stops it, and a close
or a verification cancelled while it stops its sandbox still stops it. A sandbox whose stop
failed after its seed failed is stopped again at its session's close.

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
docker build -t officeenv-sandbox:pilot resources_servers/officeenv_sandbox/sandbox
docker build -f resources_servers/officeenv_sandbox/sandbox/Dockerfile.documents \
  -t officeenv-sandbox:documents resources_servers/officeenv_sandbox/sandbox
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
The harnesses that run in the sandbox install their runtime there at session start too (Pi,
Codex, OpenClaw and OpenCode from nodejs.org, npm or GitHub, mini-SWE from PyPI, and Terminus 2
installs tmux where it is missing), so their sandboxes need the internet.

### Tools, commands or endpoints

The row's exposure decides how the harness reaches the apps (`SANDBOX_SURFACES`); the world,
the checks and the reward are the same for all three:

- `tools`: one MCP tool per op, as on the OfficeEnv server; the episode's MCP token admits its row's tools;
- `cli`: one command per app in the sandbox, `<app> <command> [--parameter=value ...]`, with
  `<app> --help` and `<app> <command> --help`, read as OfficeEnv's CLI exposure reads a command
  line. The seed writes `sandbox/app_command.sh` (POSIX sh and curl, without its comments) to
  `/usr/local/lib/company-cli/app` and links it as `/usr/local/bin/<app>` for each connected app.
  A command sends its arguments to `POST /v1/cli` with the episode's token and prints the result
  on stdout (exit 0) or the app's error on stderr (exit 1). The MCP token admits no tool, so a
  harness with nothing but a shell runs these rows (`resources_tool_transports: []`). A call to
  `/v1/cli` that does not come from an app's command (its `<app>-cli/1.0` User-Agent) is refused
  with a pointer to the command, and counted;
- `api_search`: the MCP tools `api_search` and the documentation tool find and document
  endpoints, with each endpoint's HTTP path and the curl command that calls it, and the harness
  calls `POST /v1/<app>/<op>` with the op's parameters as a JSON object. An endpoint answers with
  the app's JSON, or with its error under the error's HTTP status. The documentation tool goes by
  the name the row's exposure gives it: `api_fetch`, or `api_docs` with the exposure option
  `docs_tool` (OfficeEnv 1.11.0). The episode's MCP token admits that name, and the workspace's
  `README.md` and `sandbox/prepare_rows.py`'s prompts use it.

The file API and the content kept out of the conversation (downloads saved, pasted uploads
refused, Gmail's attachments by path) serve every exposure: a command's `--help` and the
documentation tool's answers describe the ops that move content as they behave here, as the
tools' descriptions do. The workspace's `README.md` is written for the episode's exposure, and
`sandbox_config.mcp_tool_prefix` names the MCP tools in it as the harness lists them (`company_`
for Pi). `/verify` adds `sandbox_commands`, `sandbox_commands_refused` and
`sandbox_endpoint_calls`.

Rows come from OfficeEnv's sandbox example tasks: `sandbox/prepare_rows.py` writes
`data/sandbox_pilot.jsonl` with the OfficeEnv it imports (its sandbox tasks, nine in 1.14.0). A
row holds a system and a user message and the TaskSpec, and no tool list, since the harness takes
its tools from the MCP grant. `--exposure cli` or `--exposure api_search` writes the same tasks in
that exposure, with a prompt that says how the apps are reached, to
`data/sandbox_pilot_<exposure>_prepare.jsonl`; `--tool-prefix` names the MCP tools as the harness
lists them (`company_`, Pi's, by default; `mcp__company__` for Codex).

To run the commands' rows with another harness, use its config and pass `--agent` to the
collection, for Codex CLI:

```bash
python resources_servers/officeenv_sandbox/sandbox/prepare_rows.py --exposure cli
gym env start --config resources_servers/officeenv_sandbox/configs/officeenv_sandbox_codex.yaml \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml <model and host arguments as above>
gym eval run --no-serve <the same --config and arguments> --agent codex_agent \
  --input resources_servers/officeenv_sandbox/data/sandbox_pilot_cli_prepare.jsonl --output results/codex_cli.jsonl
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

A sandbox can reach this server, so the server guards it:

- seeding, closing, verification, `/aggregate_metrics`, `/reverify_mode`, `GET /officeenv/apps`,
  `GET /task_state` and the tool routes answer only callers from this node and from
  `trusted_networks` (CIDR, empty by default), and others as routes that do not exist (404),
  before reading a request's body (a body that is not JSON is answered 422 first, naming no
  schema); no generated docs (`/openapi.json`, `/docs`) are served. A caller is on this node when
  it calls from one of the node's own addresses (a socket here binds to it), whatever the node's
  hostname resolves to (a run with `+default_host` included); a sandbox that shares the node's
  network counts as this node. Sandboxes keep the MCP endpoint, the file API and their
  exposure's routes (`/v1/cli`, `/v1/<app>/<op>`), whose tokens name their episode. When Gym's
  other servers call this one from other nodes, list their network in `trusted_networks`;
- session cookies and MCP tokens are signed with a secret drawn when the server starts
  (`get_session_secret`), not with Gym's default key, which anyone can derive from the server's
  name, so code in a sandbox cannot sign a token for another episode or for more tools. The secret
  lives in one process, so the server runs one worker (`num_workers: 1`).

The sandbox must reach this server and the model server. On a workstation the servers bind
loopback, which a container cannot reach (and a host firewall may drop connections to the host's
other addresses): set `sandbox_config.host_relay` to a host alias that reaches the host's
loopback (colima: `host.lima.internal`; Docker Desktop: `host.docker.internal`), and each sandbox
runs `sandbox/relay.py`, which forwards this server's and the model servers' ports. Relayed
connections arrive from the host's loopback, so the caller check above admits them; the signed
tokens still apply. Use the relay on a workstation only.

```bash
docker build -t officeenv-sandbox:pilot resources_servers/officeenv_sandbox/sandbox
gym env start --config resources_servers/officeenv_sandbox/configs/officeenv_sandbox.yaml \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
  --model-type inference_provider --model <model> --model-url <url> '+policy_api_key=...' \
  ++oe.resources_servers.officeenv_sandbox.sandbox_config.host_relay=host.lima.internal
gym eval run --no-serve <the same --config, --model and ++ arguments> --agent pi_agent \
  --input resources_servers/officeenv_sandbox/data/sandbox_pilot.jsonl --output results/sandbox_rollouts.jsonl
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
- the other harnesses' own names, which their Gym adapters fix: the session and runtime folders of Codex (`/tmp/nemo-gym-codex-sessions/<id>`,
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

# Licensing information
Code: this server, Apache 2.0; the `officeenv` package, NVIDIA, not yet published (its open-source release is under review)
Data: the example rows are synthetic, emitted from OfficeEnv's built-in examples

Dependencies
- nemo_gym: Apache 2.0
- officeenv: NVIDIA, not yet published; its only dependency is pydantic (MIT)
