# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""General-agent dataset environment: multi-MCP task pods with a sidecar.

Serves BOTH harness families off the same pod topology:

* Blackbox agents (claude-code / codex / mimocode) run IN the main container
  and speak MCP to the sidecar directly (streamable-http) or via the stdio
  bridge; the batch runner forwards ``mcp_servers`` / isolation kwargs to the
  agent (see run/extra/batch.py).
* The white-box DefaultAgent (RL path) runs on the trainer side — this recipe's
  ``recipes/general/env_actor.py`` (the ``DatasetEnvActor`` Ray actor) drives:

      create()  -> make_dataset_env(instance, **env["environment"]) + setup_environment()
      rollout() -> DefaultAgent(model, env=dataset_env.env, **cfg["agent"]).run(task)
      reward()  -> dataset_env.calculate_reward(timeout)

  Being out-of-pod, it reaches the sidecar MCP servers by tunnelling MCP
  JSON-RPC through ``base_env.execute()`` into ``main`` (which dials
  127.0.0.1). ``_setup_dataset_specific`` stashes the server map and the
  in-pod bridge coordinates onto the base env (``env.mcp_servers`` etc.);
  the recipe's ``mcp_proxy.discover_mcp_tools`` turns them into tools
  (see recipes/general/mcp_proxy.py, called from env_actor._create).

Pod topology (needs the two-container backend in
``recipes/general/general_agent/k8s_sidecar.py``):

* ``main``    — the agent's container (blackbox agents run here; DefaultAgent
  tools exec into here); sees ``/work/workspace`` (RW) only.
* ``sidecar`` — serves the environment's MCP tools over streamable-http
  (127.0.0.1, shared pod network namespace), owns the backing SQLite state
  (``/work/system/<mcp>/state.db``) and runs the verifier. The main container
  never mounts ``/work/system`` — DB isolation is physical, so the agent
  cannot bypass the MCP tools with direct SQL.

Instance fields (flat strings — HF ``load_dataset`` friendly):

* ``instance_id``, ``dataset_type: "general_agent"``, ``docker_image``
* ``env_task_dir`` — absolute path (on a filesystem shared with the workers)
  to a prepared task directory holding ``workspace/``, ``tools/``,
  ``system/``, the payload scripts (``sidecar_entrypoint.py``,
  ``mcp_http.py``, ``mcp_bridge.py``) and the verifier materials
  (``run_verify.py``, ``verify.py``, ``rubrics.json``, ``answer_key.json``, ...).
* ``cwd`` — should be ``/work/workspace``
* ``problem_statement`` or ``queries``
* optional ``verifier_timeout_sec``, ``agent_sessions_dir``

The yaml ``environment:`` block declares the static pod topology, which is the
same for every general_agent instance; see config/agent/general/.

Verifier materials (incl. answer keys) are uploaded to the sidecar only at
reward time, mirroring the late-upload pattern used by other datasets, so the agent can
never read them during the rollout.
"""

import json
import math
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

from mimoagent.environments.datasets.base import REWARD_TESTBED_CORRUPTED, DatasetEnvironment


SIDECAR = "sidecar"
MCP_PORT_BASE = 39101
"""First MCP port; the i-th tool (in enumeration order) serves 39101+i.
Must match sidecar_entrypoint.py's MCP_PORT_BASE."""

PAYLOAD_SCRIPTS = ("sidecar_entrypoint.py", "mcp_http.py")
"""Uploaded over the image's /installed-agent copies so the task dir's version wins."""

VERIFIER_FILES = (
    "run_verify.py",
    "verify.py",
    "rubrics.json",
    "answer_key.json",
    "verifier_meta.json",
    "_helpers.py",
)
"""Reward-time uploads (whichever exist in the task dir). answer_key.json and
friends carry ground truth — they go to the sidecar only, never to main."""


class GeneralAgentEnvironment(DatasetEnvironment):
    REPO_PATH = "/work/workspace"
    _GIT_LEAK_PREVENTION_DEFAULT = "none"
    _ANTI_HACK_CLEANUP_DEFAULT = False

    WORK_DIR = "/work"
    INSTALLED_AGENT_DIR = "/installed-agent"
    VERIFIER_LOGS_DIR = "/logs/verifier"
    AGENT_OUTPUT_DIR = "/tmp/agent_output"
    ENTRYPOINT_LOG = "/tmp/sidecar_entrypoint.log"
    MCP_STARTUP_TIMEOUT = 300
    DEFAULT_VERIFIER_TIMEOUT = 900

    AGENT_UID = 500
    SETUP_DIR = "/work/_setup"
    BRIDGE_LAUNCHER = "/work/_setup/mcp-bridge-launch"
    MCP_BRIDGE_SCRIPT = "/work/_setup/mcp_bridge.py"
    VENV_PYTHON = "/opt/openai-agents-venv/bin/python"

    AGENT_SESSIONS_POD_DIR = "/tmp/mimo-claude-logs/sessions"

    VERIFY_ENV_DEFAULTS = {"VERIFY_DETERMINISTIC": "1", "VERIFY_AGENT_JUDGE": "1"}
    VERIFY_ENV_FORWARD = (
        "VERIFY_DETERMINISTIC",
        "VERIFY_AGENT_JUDGE",
        "GA_JUDGE_KEY",
        "GA_JUDGE_URL",
        "GA_JUDGE_MODEL",
        "GA_JUDGE_EFFORT",
        "GRADE_VOTES",
        "JUDGE_API_KEY",
        "JUDGE_BASE_URL",
        "JUDGE_MODEL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "NO_PROXY",
        "no_proxy",
    )

    def __init__(self, base_env, instance: dict):
        super().__init__(base_env, instance)
        raw_task_dir = instance.get("env_task_dir") or ""
        self.task_dir = os.path.realpath(raw_task_dir) if raw_task_dir else ""
        if not self.task_dir or not os.path.isdir(self.task_dir):
            raise ValueError(f"{self.instance_id}: env_task_dir missing or not a directory: {raw_task_dir!r}")
        self.tools_dir = os.path.realpath(os.path.join(self.task_dir, "tools"))
        self.workspace_dir = os.path.realpath(os.path.join(self.task_dir, "workspace"))
        self.system_dir = os.path.realpath(os.path.join(self.task_dir, "system"))
        self.manifest = self._load_manifest()
        self.mcp_servers = {
            s["name"]: self._server_entry_from_manifest(s) for s in self.manifest.get("mcp_servers", [])
        }
        self.mcp_names = list(self.mcp_servers.keys())
        self._doer_agent: Any = None
        self._task = ""
        self._doer_result = ""

    @staticmethod
    def _enumerate_mcps(tools_dir: str) -> list[str]:
        """Sorted tool file stems.

        Must match sidecar_entrypoint.py's enumerate_tools verbatim (same sort
        key, same exclusions): the sidecar assigns port 39101+i by this exact
        order, and a divergence is a silent tool/port mix-up, not an error.
        """
        return sorted(
            p.stem
            for p in Path(tools_dir).iterdir()
            if p.is_file() and p.suffix == ".py" and not p.name.startswith("_") and p.name != "tools_test.py"
        )

    MANIFEST_FILE = "manifest.json"

    def _resolve_src(self, source: str) -> str:
        """Manifest upload sources are relative to the task dir (absolute allowed)."""
        return source if os.path.isabs(source) else os.path.join(self.task_dir, source)

    def _load_manifest(self) -> dict:
        """Read <task_dir>/manifest.json, or synthesize the default one."""
        mf = os.path.join(self.task_dir, self.MANIFEST_FILE)
        if os.path.isfile(mf):
            with open(mf, encoding="utf-8") as f:
                return self._normalize_manifest(json.load(f))
        return self._default_manifest()

    @staticmethod
    def _server_entry_from_manifest(s: dict) -> dict:
        """Translate a manifest ``mcp_servers`` entry into the SDK-native shape.

        Manifest shape (author-facing, harbor-toml-parity):
          {"name", "transport"?, "url"?, "headers"?,
           "command"?, "args"?, "env"?}

        - transport default = "streamable-http" (mini's historical default,
          harbor calls it the same). "http" / "sse" also produce a URL-based
          SDK entry — the Claude Code SDK negotiates the actual protocol at
          connect time; only "stdio" is truly distinct on the wire.
        - stdio  → {"type": "stdio", "command", "args", "env"}
        - http-family → {"type": "http", "url", "headers"}
        - sse    → {"type": "sse",  "url", "headers"}  (rare; harbor doc lists
          it as a valid transport but the 102 tasks don't use it)

        Missing fields are dropped rather than emitted as null so the SDK
        input stays minimal; the downstream ``_resolve_mcp_servers`` in the
        blackbox agent may further rewrite (isolation → sudo bridge).
        """
        transport = s.get("transport", "streamable-http")
        if transport == "stdio":
            entry: dict = {"type": "stdio", "command": s["command"]}
            if s.get("args"):
                entry["args"] = list(s["args"])
            if s.get("env"):
                entry["env"] = dict(s["env"])
            return entry
        sdk_type = "sse" if transport == "sse" else "http"
        entry = {"type": sdk_type, "url": s["url"]}
        if s.get("headers"):
            entry["headers"] = dict(s["headers"])
        return entry

    def _normalize_manifest(self, m: dict) -> dict:
        m.setdefault("cwd", self.REPO_PATH)
        m.setdefault("uploads", [])
        for u in m["uploads"]:
            u.setdefault("container", "main")
        m.setdefault("wait_ports", [])
        m.setdefault("mcp_servers", [])
        s = m.setdefault("setup", None)
        if s:
            s.setdefault("container", SIDECAR)
            s.setdefault("command", None)
            s.setdefault("timeout_sec", None)
            s.setdefault("env", None)
            if s["container"] == "main":
                raise ValueError(f"setup must not run in 'main' (agent) container; got {s!r}")
            if s.get("env"):
                for key in ("NO_PROXY", "no_proxy"):
                    if key in s["env"]:
                        self.logger.warning(
                            f"[{self.instance_id}] setup.env contains {key!r}; "
                            "this OVERRIDES pod-level NO_PROXY (mini baseline + "
                            "host_aliases auto-merge). Usually you want to "
                            "remove it — the K8s env layer handles NO_PROXY globally."
                        )
        v = m.setdefault("verifier", {})
        v.setdefault("uploads", [])
        v.setdefault("command", "python3 /work/run_verify.py")
        v.setdefault("container", SIDECAR)
        if v["container"] == "main":
            raise ValueError(f"verifier must not run in 'main' (agent) container; got {v!r}")
        v.setdefault("reward_file", f"{self.VERIFIER_LOGS_DIR}/reward.json")
        v.setdefault("reward_detail_file", f"{self.VERIFIER_LOGS_DIR}/reward_detail.json")
        v.setdefault("timeout_sec", None)
        v.setdefault("env_passthrough", None)
        v.setdefault("env", None)
        for u in m["uploads"] + v["uploads"]:
            src = self._resolve_src(u["source"])
            if not os.path.exists(src):
                raise ValueError(f"manifest source missing: {u['source']!r} (resolved {src})")
        return m

    def _default_manifest(self) -> dict:
        """Backward-compatible synthesis (no manifest.json): reproduces the
        pre-manifest behavior exactly — workspace→main, system/tools→sidecar,
        payload scripts→sidecar /installed-agent, MCP started via
        sidecar_entrypoint.py, VERIFIER_FILES late-uploaded to the sidecar, and
        run_verify.py as the reward entry."""
        have_ws = os.path.isdir(self.workspace_dir)
        have_sys = os.path.isdir(self.system_dir)
        have_tools = os.path.isdir(self.tools_dir)
        mcp_names = self._enumerate_mcps(self.tools_dir) if have_tools else []

        uploads: list[dict] = []
        if have_ws:
            uploads.append({"source": "workspace", "target": f"{self.WORK_DIR}/workspace", "container": "main"})
        if have_sys:
            uploads.append({"source": "system", "target": f"{self.WORK_DIR}/system", "container": SIDECAR})
        if have_tools:
            uploads.append({"source": "tools", "target": f"{self.WORK_DIR}/tools", "container": SIDECAR})

        setup = None
        wait_ports: list[int] = []
        mcp_servers: list[dict] = []
        if mcp_names:
            for s in PAYLOAD_SCRIPTS:
                if os.path.isfile(os.path.join(self.task_dir, s)):
                    uploads.append({"source": s, "target": f"{self.INSTALLED_AGENT_DIR}/{s}", "container": SIDECAR})
            if os.path.isfile(os.path.join(self.task_dir, "mcp_bridge.py")):
                uploads.append({"source": "mcp_bridge.py", "target": self.MCP_BRIDGE_SCRIPT, "container": "main"})
            wait_ports = [MCP_PORT_BASE + i for i in range(len(mcp_names))]
            mcp_servers = [
                {"name": n, "url": f"http://127.0.0.1:{MCP_PORT_BASE + i}/mcp"} for i, n in enumerate(mcp_names)
            ]
            setup = {
                "command": f"python3 {self.INSTALLED_AGENT_DIR}/sidecar_entrypoint.py --start-and-detach",
                "container": SIDECAR,
                "timeout_sec": self.MCP_STARTUP_TIMEOUT,
            }

        verifier = {
            "uploads": [
                {"source": f, "target": f"{self.WORK_DIR}/{f}"}
                for f in VERIFIER_FILES
                if os.path.isfile(os.path.join(self.task_dir, f))
            ],
            "command": "python3 /work/run_verify.py",
            "container": SIDECAR,
            "reward_file": f"{self.VERIFIER_LOGS_DIR}/reward.json",
            "reward_detail_file": f"{self.VERIFIER_LOGS_DIR}/reward_detail.json",
            "timeout_sec": None,
            "env_passthrough": None,
        }
        return {
            "cwd": self.REPO_PATH,
            "uploads": uploads,
            "setup": setup,
            "wait_ports": wait_ports,
            "mcp_servers": mcp_servers,
            "verifier": verifier,
        }

    def _liveness_ok(self, port: int, container: str = SIDECAR) -> dict:
        """Probe a port's LISTEN state via /proc/net/tcp (a root connect would
        be REJECTed under the mcp_isolation firewall). Pod containers share
        the network namespace, so any non-main container in the pod sees the
        same sockets — caller passes the verifier's container to guarantee it
        exists (harbor migrations may skip the SIDECAR entirely)."""
        return self.env.execute(
            'python3 -c "'
            "import sys;"
            f"hp=format({port},'04X');"
            "found=any(l.split()[3]=='0A' and l.split()[1].rsplit(':',1)[-1].upper()==hp "
            "for p in ('/proc/net/tcp','/proc/net/tcp6') if __import__('os').path.exists(p) "
            "for l in open(p).read().splitlines()[1:]);"
            'sys.exit(0 if found else 1)"',
            container=container,
            timeout=30,
        )

    def _setup_dataset_specific(self) -> None:
        env = self.env
        m = self.manifest

        for u in m["uploads"]:
            src = self._resolve_src(u["source"])
            if not os.path.exists(src):
                continue
            if u["container"] == "main":
                env.copy_to(src, u["target"], dereference=True)
            else:
                env.copy_to(src, u["target"], container=u["container"], dereference=True)

        env.mcp_servers = self.mcp_servers
        if self.mcp_servers:
            env.mcp_bridge_python = self.VENV_PYTHON
            env.mcp_bridge_script = self.MCP_BRIDGE_SCRIPT
            self._setup_privilege_isolation()

        setup = m.get("setup")
        if setup and setup.get("command"):
            cmd = setup["command"]
            if isinstance(cmd, list):
                cmd = " ".join(shlex.quote(str(x)) for x in cmd)
            env_prefix = ""
            if setup.get("env"):
                env_prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in setup["env"].items()) + " "
            container = setup["container"]
            res = env.execute(
                f"{env_prefix}{cmd} </dev/null >{self.ENTRYPOINT_LOG} 2>&1",
                cwd=self.WORK_DIR,
                timeout=int(setup.get("timeout_sec") or self.MCP_STARTUP_TIMEOUT),
                container=container,
            )
            if res.get("returncode") != 0:
                log_tail = env.execute(f"tail -c 4000 {self.ENTRYPOINT_LOG} 2>/dev/null", container=container).get(
                    "output", ""
                )
                raise RuntimeError(
                    f"{self.instance_id}: setup command failed "
                    f"(rc={res.get('returncode')}, reason={res.get('reason')}); log tail:\n{log_tail}"
                )

        if self.mcp_servers:
            self.logger.info(
                f"{self.instance_id}: {len(self.mcp_servers)} MCP server(s) ready: {', '.join(self.mcp_names)}"
            )
        else:
            self.logger.info(f"{self.instance_id}: MCP-free task (no servers)")

    @property
    def _mcp_isolation(self) -> dict | None:
        return getattr(getattr(self.env, "config", None), "mcp_isolation", None)

    def agent_isolation_kwargs(self) -> dict:
        """Config the blackbox agent needs to honor the firewall (empty when
        isolation is off). batch.py forwards these to the agent (like
        mcp_servers) so codex/claude drop to agent_uid and launch the bridge
        via ``sudo -u <bridge_user> <launcher>``."""
        iso = self._mcp_isolation
        if not iso:
            return {}
        return {
            "agent_uid": iso.get("agent_uid", self.AGENT_UID),
            "mcp_bridge_user": iso.get("bridge_user", "omni"),
            "mcp_bridge_launcher": self.BRIDGE_LAUNCHER,
        }

    def _setup_privilege_isolation(self) -> None:
        iso = self._mcp_isolation
        if not iso:
            return
        bridge_user = iso.get("bridge_user", "omni")
        agent_uid = iso.get("agent_uid", self.AGENT_UID)
        cases = "\n".join(f'  {name}) url="{srv["url"]}";;' for name, srv in self.mcp_servers.items())
        setup = f"""
set -e
id -u agentu 2>/dev/null || useradd -u {agent_uid} -m -s /bin/bash agentu
mkdir -p {self.SETUP_DIR}
cat > {self.BRIDGE_LAUNCHER} <<'LAUNCH'
#!/bin/bash
name=""
while [ $# -gt 0 ]; do case "$1" in --name) name="$2"; shift 2;; *) shift;; esac; done
case "$name" in
{cases}
  *) echo "unknown mcp: $name" >&2; exit 1;;
esac
exec env MCP_ENDPOINT_URL="$url" {self.VENV_PYTHON} {self.SETUP_DIR}/mcp_bridge.py --name "$name"
LAUNCH
chown {bridge_user} {self.BRIDGE_LAUNCHER}
chmod 0700 {self.BRIDGE_LAUNCHER}
echo 'agentu ALL=({bridge_user}) NOPASSWD: {self.BRIDGE_LAUNCHER} *' > /etc/sudoers.d/mcp-bridge
chmod 0440 /etc/sudoers.d/mcp-bridge
visudo -c >/dev/null
chown -R {agent_uid}:{agent_uid} {self.WORK_DIR}/workspace 2>/dev/null || true
"""
        res = self.env.execute(setup)  # main container, root
        if res.get("returncode") != 0:
            raise RuntimeError(
                f"{self.instance_id}: privilege-isolation setup failed "
                f"(rc={res.get('returncode')}): {res.get('output', '')[:400]}"
            )
        self.logger.info(
            f"{self.instance_id}: privilege isolation ready "
            f"(agent uid={agent_uid}, bridge user={bridge_user}, url hidden from agent)"
        )

    def _capture_model_diff(self) -> tuple[str, str]:
        return "", ""

    def attach_rollout(self, *, agent: Any = None, task: str = "", result: str = "") -> None:
        """Runner hook (between agent.run() and calculate_reward()).

        All parts optional — recalc mode has no agent; the verifier then works
        from the DB post-state and whatever answer.md the agent left behind.
        """
        self._doer_agent = agent
        self._task = task or ""
        self._doer_result = result or ""

    def _do_calculate_reward(
        self, timeout: int | float | None = None, model_patch: str = ""
    ) -> tuple[float, str, dict]:
        env = self.env

        if not os.environ.get("GA_JUDGE_KEY"):
            _key_file = os.environ.get("GA_JUDGE_KEY_FILE")
            if _key_file and os.path.isfile(_key_file):
                try:
                    import yaml as _y

                    _judge = ((_y.safe_load(open(_key_file)) or {}).get("channels") or {}).get("judge") or {}
                    if _judge.get("api_key"):
                        os.environ["GA_JUDGE_KEY"] = _judge["api_key"]
                        os.environ["VERIFY_AGENT_JUDGE"] = "1"
                        if _judge.get("model_name"):
                            os.environ.setdefault("GA_JUDGE_MODEL", _judge["model_name"])
                        if _judge.get("base_url"):
                            os.environ.setdefault("GA_JUDGE_URL", _judge["base_url"])
                except Exception as _e:
                    self.logger.warning("GA_JUDGE_KEY_FILE %s unreadable: %s", _key_file, _e)

        if not os.environ.get("GA_JUDGE_KEY"):
            return (
                0.0,
                "GA_JUDGE_KEY missing from controller env "
                "(set it, or point GA_JUDGE_KEY_FILE at a yaml with channels.judge.api_key)",
                {
                    "error_category": REWARD_TESTBED_CORRUPTED,
                    "reward_error": "judge_key_missing",
                },
            )

        probe_container = self.manifest["verifier"]["container"]
        for wp in self.manifest.get("wait_ports") or []:
            probe = self._liveness_ok(int(wp), container=probe_container)
            if probe.get("returncode") != 0:
                return (
                    0.0,
                    probe.get("output", ""),
                    {
                        "error_category": REWARD_TESTBED_CORRUPTED,
                        "reward_error": "mcp_backend_down",
                        "dead_port": int(wp),
                    },
                )

        v = self.manifest["verifier"]
        toks = shlex.split(v.get("command", ""))
        entry = next((t for t in toks if t.startswith(f"{self.WORK_DIR}/") and t.endswith((".py", ".sh"))), None)
        if entry:
            upload_targets = {u.get("target") for u in v.get("uploads", [])}
            if entry not in upload_targets and not os.path.isfile(
                os.path.join(self.task_dir, os.path.basename(entry))
            ):
                return (
                    0.0,
                    "",
                    {
                        "error_category": REWARD_TESTBED_CORRUPTED,
                        "reward_error": "missing_run_verify",
                        "missing_script": entry,
                    },
                )
        vcontainer = v["container"]
        for u in v["uploads"]:
            src = self._resolve_src(u["source"])
            if os.path.exists(src):
                env.copy_to(src, u["target"], container=vcontainer, dereference=True)

        self._inject_agent_sessions(vcontainer)
        self._write_answer_md()

        reward_file = v["reward_file"]
        detail_file = v.get("reward_detail_file")
        env.execute(
            f"mkdir -p {os.path.dirname(reward_file)} {self.AGENT_OUTPUT_DIR} && "
            f"rm -f {reward_file} {detail_file or ''}",
            container=vcontainer,
        )
        exec_timeout = (
            int(timeout)
            if timeout is not None
            else int(
                v.get("timeout_sec") or self.instance.get("verifier_timeout_sec") or self.DEFAULT_VERIFIER_TIMEOUT
            )
        )
        run_res = env.execute(
            f"{self._verify_env_prefix(v.get('env_passthrough'), v.get('env'))} {v['command']}",
            cwd=self.WORK_DIR,
            timeout=exec_timeout,
            container=vcontainer,
        )
        output = run_res.get("output", "")
        verifier_rc = run_res.get("returncode", 1)

        reward_raw = env.execute(f"cat {reward_file} 2>/dev/null", container=vcontainer).get("output", "").strip()
        detail_raw = ""
        if detail_file:
            detail_raw = env.execute(f"cat {detail_file} 2>/dev/null", container=vcontainer).get("output", "").strip()

        def _mask(reward_error: str, **kw):
            info = {
                "error_category": REWARD_TESTBED_CORRUPTED,
                "reward_error": reward_error,
                "verifier_returncode": verifier_rc,
                **kw,
            }
            try:
                d = json.loads(detail_raw)
                if isinstance(d, dict) and d.get("reward_error"):
                    info["verifier_reward_error"] = d["reward_error"]
            except Exception:
                pass
            return 0.0, output, info

        try:
            reward = float(json.loads(reward_raw)["reward"])
        except Exception:
            return _mask("missing_or_invalid_reward_json", reward_json_raw=reward_raw[:400])
        if not math.isfinite(reward) or not (0.0 <= reward <= 1.0):
            return _mask("reward_out_of_range", reward_json_raw=reward_raw[:400])

        extra: dict[str, Any] = {"verifier_returncode": verifier_rc}
        if self.instance.get("reward_binary_fullscore"):
            extra["raw_reward"] = reward
            reward = 1.0 if reward >= 1.0 - 1e-6 else 0.0
        if detail_raw:
            try:
                extra["reward_detail"] = json.loads(detail_raw)
                self._annotate_rubric_kinds(extra["reward_detail"])
            except Exception:
                extra["reward_detail_raw"] = detail_raw[:2000]
        if os.environ.get("GA_PULL_POST_STATE") == "1":
            root = os.environ.get("GA_POST_STATE_DIR", "")
            if root:
                pod = getattr(env, "pod_name", "") or "pod"
                self.pull_post_state(os.path.join(root, self.instance_id, pod))
        return reward, output, extra

    def _annotate_rubric_kinds(self, detail: dict) -> None:
        """Join each rubric result back to its verifier_meta.json definition.

        Adds method (rule/llm) / weight / gate / dim to the entries in ``results``.
        Display-only: the score is not touched. ``src_protect`` is verify.py's own
        source-conservation gate and has no item in the meta file.
        """
        try:
            with open(os.path.join(self.task_dir, "verifier_meta.json"), encoding="utf-8") as f:
                items = {it.get("id"): it for it in (json.load(f).get("items") or [])}
        except Exception:
            return
        for r in detail.get("results") or []:
            it = items.get(r.get("id"))
            if it:
                r["method"] = it.get("method")
                r["weight"] = it.get("weight", 1)
                r["gate"] = bool(it.get("gate"))
                r["dim"] = it.get("dim")
            elif r.get("id") == "src_protect":
                r["method"] = "gate:auto"
                r["gate"] = True

    def pull_post_state(self, dest_dir: str) -> None:
        """Snapshot the post-run environment state into ``dest_dir`` as two
        tarballs: workspace/ from main (the RW side — includes the
        materialized answer.md) and system/ (per-MCP state.db) from the
        sidecar. Enables re-running verify offline afterwards (judge outage
        at reward time, audits, re-scoring) — see
        scripts/reverify_post_state.py in general-agent. Best-effort per
        part: a failed pull logs a warning and never affects the reward
        already computed. Invoked by the batch runner after
        calculate_reward() when GA_PULL_POST_STATE=1."""
        os.makedirs(dest_dir, exist_ok=True)
        _want = {p.strip() for p in os.environ.get("GA_POST_STATE_PARTS", "workspace,system").split(",") if p.strip()}
        cwd = (self.manifest.get("cwd") or self.REPO_PATH).rstrip("/") or "/"
        ws_srcdir = os.path.dirname(cwd) or "/"
        ws_srcbase = os.path.basename(cwd) or "workspace"
        import tarfile

        for name, container, srcdir, srcbase in (
            ("workspace", None, ws_srcdir, ws_srcbase),
            ("system", SIDECAR, self.WORK_DIR, "system"),
        ):
            if _want and name not in _want:
                continue
            pod_tar = f"/tmp/post_state_{name}.tar.gz"
            try:
                res = self.env.execute(
                    f"tar -czf {pod_tar} -C {shlex.quote(srcdir)} {shlex.quote(srcbase)}",
                    timeout=600,
                    container=container,
                )
                if res.get("returncode") != 0:
                    raise RuntimeError((res.get("output") or "")[:200])
                local_tar = os.path.join(dest_dir, f"{name}.tar.gz")
                self.env.copy_out(pod_tar, local_tar, container=container)
                with tarfile.open(local_tar) as tf:
                    tf.extractall(dest_dir)
                os.remove(local_tar)
                extracted = os.path.join(dest_dir, srcbase)
                target = os.path.join(dest_dir, name)
                if srcbase != name and os.path.isdir(extracted):
                    if os.path.isdir(target):
                        import shutil

                        shutil.rmtree(target)
                    os.replace(extracted, target)
            except Exception as e:
                self.logger.warning(f"{self.instance_id}: post-state pull of {name} failed ({e!r})")

    def _verify_env_prefix(
        self,
        passthrough: list[str] | None = None,
        explicit: dict[str, str] | None = None,
    ) -> str:
        env = dict(self.VERIFY_ENV_DEFAULTS)
        for key in passthrough or self.VERIFY_ENV_FORWARD:
            value = os.environ.get(key)
            if value:
                env[key] = value
        if explicit:
            env.update({k: str(v) for k, v in explicit.items()})
        return " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())

    def _inject_agent_sessions(self, container: str = SIDECAR) -> None:
        """main:/tmp/mimo-claude-logs/sessions → <verifier container>:/tmp/agent_output/sessions.

        Two hops via the controller — there is no shared volume between the
        session dir and the verifier container (deliberately: /work/workspace
        is part of the graded state and must not be polluted with logs).
        Missing sessions are tolerated: run_verify.py falls back to
        workspace/answer.md.
        """
        sessions_dir = self.instance.get("agent_sessions_dir") or self.AGENT_SESSIONS_POD_DIR
        if not hasattr(self.env, "copy_out"):
            return
        check = self.env.execute(f"test -d {shlex.quote(sessions_dir)}")
        if check.get("returncode") != 0:
            self.logger.info(f"{self.instance_id}: no agent sessions at {sessions_dir}, skipping")
            return
        with tempfile.TemporaryDirectory(prefix="ga_sessions_") as tmp:
            local = os.path.join(tmp, "sessions")
            try:
                self.env.copy_out(sessions_dir, local)
                self.env.copy_to(local, f"{self.AGENT_OUTPUT_DIR}/sessions", container=container)
            except Exception as e:
                self.logger.warning(
                    f"{self.instance_id}: session injection failed ({e!r}); verifier will fall back to answer.md"
                )

    def _write_answer_md(self) -> None:
        """Persist the doer's final reply as workspace/answer.md (via main —
        the sidecar mounts the workspace read-only). An answer.md the agent
        wrote itself wins."""
        if not self._doer_result:
            return
        check = self.env.execute(f"test -s {self.REPO_PATH}/answer.md")
        if check.get("returncode") == 0:
            return
        try:
            self.copy_text_to(self._doer_result, f"{self.REPO_PATH}/answer.md")
        except Exception as e:
            self.logger.warning(f"{self.instance_id}: writing answer.md failed ({e!r})")
